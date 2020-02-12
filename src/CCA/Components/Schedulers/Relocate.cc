/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <CCA/Components/Schedulers/Relocate.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Containers/Array2.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/ProgressiveWarning.h>

#include <map>
#include <set>

#define RELOCATE_TAG            0x3fff

using namespace Uintah;

namespace Uintah {
  extern Dout g_mpi_dbg;
}

namespace {
  Dout g_total_reloc("RELOCATE_SCATTER_DBG", "Schedulers", "prints info on particle scatter ops", false);

  DebugStream coutdbg("RELOCATE_DBG", "Schedulers", "prints particle relocation neighbor patches", false);
}

Relocate::~Relocate()
{
  if(reloc_matls && reloc_matls->removeReference()) {
    delete reloc_matls;
  }
  for (size_t p = 0; p < destroyMe_.size(); p++) {
    VarLabel::destroy(destroyMe_[p]);
  }
}

namespace Uintah {
  struct ScatterRecord {
    const Patch* fromPatch;
    const Patch* toPatch;    
    IntVector vectorToNeighbor;
    int matl;
    int levelIndex;
    ParticleSubset* send_pset;
    
    ScatterRecord(const Patch* fromPatch, const Patch* toPatch, int matl, int levelIndex)
      : fromPatch(fromPatch), toPatch(toPatch), matl(matl), levelIndex(levelIndex), send_pset(0)
    {
      ASSERT(fromPatch != 0);
      ASSERT(toPatch != 0);
      
      vectorToNeighbor = toPatch->getExtraCellLowIndex() - fromPatch->getExtraCellLowIndex();
    }

    // Note that when the ScatterRecord going from a real patch to
    // a virtual patch has an equivalent representation going from
    // a virtual patch to a real patch (wrap-around, periodic bound. cond.).
    bool equivalent(const ScatterRecord& sr){ 
      return (toPatch->getRealPatch() == sr.toPatch->getRealPatch()) && (matl == sr.matl) && (vectorToNeighbor == sr.vectorToNeighbor);
    }
  };


  std::ostream& operator<<(std::ostream& out, const ScatterRecord & r){
    out.setf(std::ios::scientific, std::ios::floatfield);
    out.precision(4);
    out << " Scatter Record, matl: " << r.matl
        << " Level: " << r.levelIndex
        << " numParticles " << r.send_pset->numParticles()
        << " (Particle moving from Patch " << r.fromPatch->getID() 
        << ", to Patch " <<  r.toPatch->getID() << ")"
        << " vectorToNeighbor " << r.vectorToNeighbor;
    out.setf(std::ios::scientific, std::ios::floatfield);
    return out;
  }


  typedef std::multimap<std::pair<const Patch*, int>, ScatterRecord*> maptype;

#if 0 // Not used???
  struct CompareScatterRecord {
  
    bool operator()(const ScatterRecord* sr1, const ScatterRecord* sr2) const
    {
      return 
      ((sr1->toPatch->getRealPatch() != sr2->toPatch->getRealPatch()) ?
       (sr1->toPatch->getRealPatch() <  sr2->toPatch->getRealPatch()) :
       ((sr1->matl != sr2->matl)     ? (sr1->matl < sr2->matl) :
    compareIntVector(sr1->vectorToNeighbor, sr2->vectorToNeighbor)));
    }
    
    bool compareIntVector(const IntVector& v1, const IntVector& v2) const
    {
      return (v1.x() != v2.x()) ? (v1.x() < v2.x()) :
            ((v1.y() != v2.y()) ? (v1.y() < v2.y()) : 
             (v1.z() < v2.z()));
    }    
  };
#endif

  typedef std::vector<const Patch*> patchestype;

  struct MPIScatterProcessorRecord {
    patchestype patches;
    void sortPatches();
  };

  typedef std::map<int, MPIScatterProcessorRecord*> procmaptype;


  struct MPIRecvBuffer {
    MPIRecvBuffer* next;
    char* databuf;
    int bufsize;
    int numParticles;
    MPIRecvBuffer(char* databuf, int bufsize, int numParticles)
      : next(0), databuf(databuf), bufsize(bufsize), numParticles(numParticles)
    {
    }
  };
  typedef std::map<std::pair<const Patch*, int>, MPIRecvBuffer*> recvmaptype;

  class MPIScatterRecords {
  public:
    // map the to patch and matl to the ScatterRecord
    maptype records;
    
    procmaptype procs;

    ScatterRecord* findOrInsertRecord(const Patch* from, const Patch* to, int matl, int curLevelIndex, ParticleSubset* pset);
    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl, int curLevelIndex);
    
    void addNeighbor( LoadBalancer * lb, const ProcessorGroup * pg, const Patch * to );

    recvmaptype recvs;
    void saveRecv(const Patch* to, int matl, char* databuf, int bufsize, int numParticles);
    MPIRecvBuffer* findRecv(const Patch* to, int matl);

    ~MPIScatterRecords();
  };
} // End namespace Uintah


//______________________________________________________________________
// All variables that will be relocated must exist on the same levels.
// You cannot have one variable that exists on a different number of levels
void
Relocate::scheduleParticleRelocation(       Scheduler                                  * sched,
                                      const ProcessorGroup                             * pg,
                                            LoadBalancer                           * lb,
                                      const LevelP                                     & coarsestLevelwithParticles,
                                      const VarLabel                                   * posLabel,
                                      const std::vector<std::vector<const VarLabel*> > & otherLabels,
                                      const MaterialSet                                * matls )
{
  //In this version of the relocation algorithm, the user provides a list of varlabels that require
  // relocation. Uintah will create a mirror list of temporary, post-reloc variables. Uintah will
  // then fill those post-reloc variables with relocated variables. After all communication and
  // relocation is completed, Uintah will then copy the relocated variables back into the original
  // variables that the user has provided
  
  // create the post relocation position variables
  VarLabel* posPostRelocLabel = VarLabel::create(posLabel->getName() + "+", posLabel->typeDescription());
  destroyMe_.push_back(posPostRelocLabel);
  
  // create a vector of post relocation variables
  std::vector<std::vector<const VarLabel*> > postRelocOtherLabels;
  // fill the list of post relocation variables
  for (size_t m = 0; m < otherLabels.size(); m++) {
    std::vector<const VarLabel*> tmp;
    postRelocOtherLabels.push_back(tmp);
    for (size_t p = 0; p < otherLabels[m].size(); p++) {
      const VarLabel* pVarLabel = otherLabels[m][p];
      const std::string pPostRelocVarName = pVarLabel->getName() + "+";
      const VarLabel* pPostRelocVarLabel;
      if (VarLabel::find(pPostRelocVarName)) {
        pPostRelocVarLabel = VarLabel::find(pPostRelocVarName);
      } else {
        pPostRelocVarLabel = VarLabel::create(pPostRelocVarName, pVarLabel->typeDescription());
        destroyMe_.push_back(pPostRelocVarLabel);
      }
      postRelocOtherLabels[m].push_back(pPostRelocVarLabel);
    }
  }
  // Only allow particles at the finest level for now
  //  if(level->getIndex() != level->getGrid()->numLevels()-1)
  //    return;
  reloc_old_posLabel = posLabel;
  reloc_old_labels   = otherLabels;
  reloc_new_posLabel = posPostRelocLabel;
  reloc_new_labels   = postRelocOtherLabels;
  
  if(reloc_matls && reloc_matls->removeReference()){
    delete reloc_matls;
  }
  
  reloc_matls = matls;
  reloc_matls->addReference();
  ASSERTEQ(reloc_old_labels.size(), reloc_new_labels.size());
  int numMatls = (int)reloc_old_labels.size();
  ASSERTEQ(matls->size(), 1);
  
  // be careful with matls - we need to access reloc_labels linearly, but
  // they may not be in consecutive order - so get the matl from the matl
  // subset whenever you schedule a task or use the dw.
  const MaterialSubset* matlsub = matls->getSubset(0);
  ASSERTEQ(numMatls, matlsub->size());
  
  for (int m = 0; m< numMatls; m++){
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());
  }
  Task* t = scinew Task("Relocate::relocateParticles",
                        this, &Relocate::relocateParticlesModifies, coarsestLevelwithParticles.get_rep());
  if( lb ) {
    t->usesMPI( true );
  }
  t->requires( Task::NewDW, reloc_old_posLabel, Ghost::None);
  //t->modifies( reloc_old_posLabel );
  
  for(int m=0;m < numMatls;m++){
    MaterialSubset* thismatl = scinew MaterialSubset();
    thismatl->add(matlsub->get(m));
    
    for(int i=0;i<(int)reloc_old_labels[m].size();i++){
      t->requires( Task::NewDW, reloc_old_labels[m][i], Ghost::None);
//      t->modifies( reloc_old_labels[m][i] );
    }
    
    t->computes( reloc_new_posLabel, thismatl);
    for(int i=0;i<(int)reloc_new_labels[m].size();i++){
      t->computes(reloc_new_labels[m][i], thismatl);
    }
  }
  
  PatchSet * patches;
  if( !coarsestLevelwithParticles->hasFinerLevel() ) {
    // only case since the below version isn't const
    patches = const_cast< PatchSet * >( lb->getPerProcessorPatchSet(coarsestLevelwithParticles) );
  }
  else {
    GridP grid = coarsestLevelwithParticles->getGrid();
    // make per-proc patch set of each level >= level
    patches = scinew PatchSet();
    patches->createEmptySubsets(pg->nRanks());
    
    for (int i = coarsestLevelwithParticles->getIndex(); i < grid->numLevels(); i++) {
      
      const PatchSet* p = lb->getPerProcessorPatchSet(grid->getLevel(i));
      
      for (int proc = 0; proc < pg->nRanks(); proc++) {
        for (int j = 0; j < p->getSubset(proc)->size(); j++) {
          const Patch* patch = p->getSubset(proc)->get(j);
          patches->getSubset(lb->getPatchwiseProcessorAssignment(patch))->add(patch);
        }
      }
    }
  }

  printSchedule(patches,coutdbg,"Relocate::scheduleRelocateParticles");

  t->setType(Task::OncePerProc);
  sched->addTask(t, patches, matls);
  m_lb = lb;
}

//______________________________________________________________________
// All variables that will be relocated must exist on the same levels. 
// You cannot have one variable that exists on a different number of levels  
void
Relocate::scheduleParticleRelocation( Scheduler                                        * sched,
                                      const ProcessorGroup                             * pg,
                                      LoadBalancer                                 * lb,
                                      const LevelP                                     & coarsestLevelwithParticles,
                                      const VarLabel                                   * old_posLabel,
                                      const std::vector<std::vector<const VarLabel*> > & old_labels,
                                      const VarLabel                                   * new_posLabel,
                                      const std::vector<std::vector<const VarLabel*> > & new_labels,
                                      const VarLabel                                   * particleIDLabel,
                                      const MaterialSet                                * matls )
{
// Only allow particles at the finest level for now
//  if(level->getIndex() != level->getGrid()->numLevels()-1)
//    return;
  reloc_old_posLabel = old_posLabel;
  reloc_old_labels   = old_labels;
  reloc_new_posLabel = new_posLabel;
  reloc_new_labels   = new_labels;
  particleIDLabel_   = particleIDLabel;
  
  if(reloc_matls && reloc_matls->removeReference()){
    delete reloc_matls;
  }
    
  reloc_matls = matls;
  reloc_matls->addReference();
  ASSERTEQ(reloc_old_labels.size(), reloc_new_labels.size());
  int numMatls = (int)reloc_old_labels.size();
  ASSERTEQ(matls->size(), 1);

  // be careful with matls - we need to access reloc_labels linearly, but
  // they may not be in consecutive order - so get the matl from the matl
  // subset whenever you schedule a task or use the dw.
  const MaterialSubset* matlsub = matls->getSubset(0);
  ASSERTEQ(numMatls, matlsub->size());
  
  for (int m = 0; m< numMatls; m++){
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());
  }
  Task* t = scinew Task("Relocate::relocateParticles",
                  this, &Relocate::relocateParticles, coarsestLevelwithParticles.get_rep());
  if(lb){
    t->usesMPI(true);
  }
  t->requires( Task::NewDW, old_posLabel, Ghost::None);
  
  for(int m=0;m < numMatls;m++){
    MaterialSubset* thismatl = scinew MaterialSubset();
    thismatl->add(matlsub->get(m));
    
    for(int i=0;i<(int)old_labels[m].size();i++){
      t->requires( Task::NewDW, old_labels[m][i], thismatl, Ghost::None);
    }
    
    t->computes( new_posLabel, thismatl);
    for(int i=0;i<(int)new_labels[m].size();i++){
      t->computes(new_labels[m][i], thismatl);
    }
  }
  
  PatchSet* patches;
  if(!coarsestLevelwithParticles->hasFinerLevel()){
    // only case since the below version isn't const
    patches = const_cast<PatchSet*>(lb->getPerProcessorPatchSet(coarsestLevelwithParticles)); 
  }else {
    GridP grid = coarsestLevelwithParticles->getGrid();
    // make per-proc patch set of each level >= level
    patches = scinew PatchSet();
    patches->createEmptySubsets(pg->nRanks());
   
    for (int i = coarsestLevelwithParticles->getIndex(); i < grid->numLevels(); i++) {
      
      const PatchSet* p = lb->getPerProcessorPatchSet(grid->getLevel(i));
      
      for (int proc = 0; proc < pg->nRanks(); proc++) {
        for (int j = 0; j < p->getSubset(proc)->size(); j++) {
          const Patch* patch = p->getSubset(proc)->get(j);
          patches->getSubset(lb->getPatchwiseProcessorAssignment(patch))->add(patch);
        }
      }
    }
  }
  
  
  printSchedule(patches,coutdbg,"Relocate::scheduleRelocateParticles");
  
  t->setType(Task::OncePerProc);
  sched->addTask(t, patches, matls);
  m_lb = lb;
}
//______________________________________________________________________
//
void MPIScatterRecords::saveRecv(const Patch* to,
                                 int matl,
                                 char* databuf,
                                 int datasize,
                                 int numParticles)
{
  recvmaptype::key_type key(to, matl);
  recvmaptype::iterator iter = recvs.find(key);
  MPIRecvBuffer* record = scinew MPIRecvBuffer(databuf, datasize, numParticles);
  
  if(iter == recvs.end()){
    recvs[key]=record;
  } else {
    record->next=iter->second;
    recvs[key]=record;
  }
}
//______________________________________________________________________
//
MPIRecvBuffer* MPIScatterRecords::findRecv(const Patch* to, int matl)
{
  recvmaptype::iterator iter = recvs.find(std::make_pair(to, matl));
  if(iter == recvs.end()){
    return 0;
  }else{
    return iter->second;
  }
}
//______________________________________________________________________
//
ScatterRecord* MPIScatterRecords::findOrInsertRecord(const Patch* fromPatch,
                                                     const Patch* toPatch, 
                                                     int matl,
                                                     int curLevelIndex,
                                                     ParticleSubset* pset)
{
  ASSERT(toPatch != 0);
  IntVector vectorToNeighbor = toPatch->getExtraCellLowIndex() - fromPatch->getExtraCellLowIndex();
  const Patch* realToPatch = toPatch->getRealPatch();
  
  std::pair<maptype::iterator, maptype::iterator> pr = records.equal_range(std::make_pair(realToPatch, matl));
  
  //__________________________________
  // loop over all scatter records
  // Does this record exist if so return.
  for(;pr.first != pr.second;pr.first++){
    ScatterRecord* sr = pr.first->second;

    if((realToPatch == sr->toPatch->getRealPatch()) && 
       (curLevelIndex == sr->levelIndex) && 
       (vectorToNeighbor == sr->vectorToNeighbor) ){
      return sr;
    }
  }
  
  //__________________________________
  //  Create a new record and insert it into
  //  all records
  ScatterRecord* rec = scinew ScatterRecord(fromPatch, toPatch, matl, curLevelIndex);
  rec->send_pset = scinew ParticleSubset(0, -1, 0);
  records.insert(maptype::value_type(std::make_pair(realToPatch, matl), rec));
  return rec;
}
//______________________________________________________________________
//
ScatterRecord* MPIScatterRecords::findRecord(const Patch* fromPatch,
                                             const Patch* toPatch,
                                             int matl, 
                                             int curLevelIndex)
{
  ASSERT(toPatch != 0);
  
  IntVector vectorToNeighbor = toPatch->getExtraCellLowIndex() - fromPatch->getExtraCellLowIndex();
  const Patch* realToPatch = toPatch->getRealPatch();
  const Patch* realFromPatch = fromPatch->getRealPatch();

  std::pair<maptype::iterator, maptype::iterator> pr = records.equal_range(std::make_pair(realToPatch, matl));
  
  //__________________________________
  // loop over all scatter records
  // Does this record exist if so return.
  for(;pr.first != pr.second;pr.first++){
    ScatterRecord* sr = pr.first->second;

    if((realToPatch      == sr->toPatch->getRealPatch()) && 
       (curLevelIndex    == sr->levelIndex) && 
       (matl             == sr->matl) &&
       (vectorToNeighbor == sr->vectorToNeighbor) &&
       (realFromPatch    == sr->fromPatch->getRealPatch()) &&
       (fromPatch        != toPatch)) {
      return sr;
    }
  }
  
  return 0;
}
//______________________________________________________________________
//
static bool ComparePatches(const Patch* p1, const Patch* p2)
{
  return p1->getID() < p2->getID();
}
//______________________________________________________________________
//
void MPIScatterProcessorRecord::sortPatches()
{
  std::sort(patches.begin(), patches.end(), ComparePatches);
}
//______________________________________________________________________
//
void MPIScatterRecords::addNeighbor( LoadBalancer     * lb, 
                                     const ProcessorGroup * pg,
                                     const Patch          * neighbor )
{
  neighbor = neighbor->getRealPatch();
  int toProc = lb->getPatchwiseProcessorAssignment(neighbor);
  ASSERTRANGE(toProc, 0, pg->nRanks());
  
  procmaptype::iterator iter = procs.find(toProc);
  
  if(iter == procs.end()){
    MPIScatterProcessorRecord* pr = scinew MPIScatterProcessorRecord();
    procs[toProc]=pr;
    pr->patches.push_back(neighbor);
  } else {
    // This is linear, with the hope that the number of patches per
    // processor will not be huge.
    MPIScatterProcessorRecord* pr = iter->second;
    int i;
    for(i=0;i<(int)pr->patches.size();i++){
      if(pr->patches[i] == neighbor){
        break;
      }
    }
    if(i==(int)pr->patches.size()){
      pr->patches.push_back(neighbor);
    }
  }
}
//______________________________________________________________________
//
MPIScatterRecords::~MPIScatterRecords()
{
  for(procmaptype::iterator iter = procs.begin(); iter != procs.end(); iter++){
    delete iter->second;
  }
    
  for(maptype::iterator mapiter = records.begin(); mapiter != records.end();mapiter++){
    delete mapiter->second->send_pset;    
    delete mapiter->second;
  }
  
  for(recvmaptype::iterator iter = recvs.begin(); iter != recvs.end(); iter++){
    MPIRecvBuffer* p = iter->second;
  
    while(p){
      MPIRecvBuffer* next = p->next;
      delete p;
      p=next;
    }
    
  }
}
//______________________________________________________________________
//
void
Relocate::exchangeParticles(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw, 
                            MPIScatterRecords* scatter_records,
                            int total_reloc[3])
{
  // this level is the coarsest level involved in the relocation
  const Level* coarsestLevel = patches->get(0)->getLevel();
  GridP grid = coarsestLevel->getGrid();
  
  int numMatls = (int)reloc_old_labels.size();

  int me = pg->myRank();
  for(procmaptype::iterator iter = scatter_records->procs.begin();
                           iter != scatter_records->procs.end(); iter++){
    
    if(iter->first == me){
      continue;   // Local
    }

    MPIScatterProcessorRecord* procRecord = iter->second;
    procRecord->sortPatches();

    // Go through once to calc the size of the message
    int psize;
    Uintah::MPI::Pack_size(1, MPI_INT, pg->getComm(), &psize);
    int sendsize  = psize; // One for the count of active patches
    int numactive = 0;
    std::vector<int> datasizes;
    
    for(patchestype::iterator it = procRecord->patches.begin(); it != procRecord->patches.end(); it++){
      const Patch* toPatch = *it;
      
      for(int m=0;m<numMatls;m++){
        int matl = matls->get(m);

        int numVars = (int)reloc_old_labels[m].size();
        int numParticles = 0;
        
        std::pair<maptype::iterator, maptype::iterator> pr;
        pr = scatter_records->records.equal_range(std::make_pair(toPatch, matl));
  
        for(;pr.first != pr.second; pr.first++){
          numactive++;
          int psize;
          
          Uintah::MPI::Pack_size(4, MPI_INT, pg->getComm(), &psize);
          sendsize += psize; // Patch ID, matl #, # particles, datasize
          int orig_sendsize = sendsize;
          ScatterRecord* record = pr.first->second;
          int np = record->send_pset->numParticles();
          
          numParticles += np;
          ParticleSubset* pset         = old_dw->getParticleSubset(matl, record->fromPatch);
          ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
          ParticleSubset* send_pset    = record->send_pset;
          
          posvar->packsizeMPI(&sendsize, pg, send_pset);
          
          for(int v=0;v<numVars;v++){
            ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);
            var->packsizeMPI(&sendsize, pg, send_pset);
          }
          int datasize = sendsize-orig_sendsize;
          datasizes.push_back(datasize);
        }
      }  // matl loop
    } // patches loop
    
    
    // Create the buffer for this message
    char* buf = scinew char[sendsize];
    int position=0;

    // And go through it again to pack the message
    int idx=0;
    Uintah::MPI::Pack(&numactive, 1, MPI_INT, buf, sendsize, &position, pg->getComm());
    
    for(patchestype::iterator it = procRecord->patches.begin();it != procRecord->patches.end(); it++){
      const Patch* toPatch = *it;
      
      for(int m=0;m<numMatls;m++){
        int matl = matls->get(m);
        int numVars = (int)reloc_old_labels[m].size();

        std::pair<maptype::iterator, maptype::iterator> pr;
        pr = scatter_records->records.equal_range(std::make_pair(toPatch, matl));
  
        for(;pr.first != pr.second; pr.first++){
          int patchid = toPatch->getID();
          Uintah::MPI::Pack(&patchid, 1, MPI_INT, buf, sendsize, &position, pg->getComm());
          Uintah::MPI::Pack(&matl,    1, MPI_INT, buf, sendsize, &position, pg->getComm());
          
          ScatterRecord* record = pr.first->second;
          int totalParticles = record->send_pset->numParticles();
          
          Uintah::MPI::Pack(&totalParticles, 1, MPI_INT, buf, sendsize, &position, pg->getComm());
          
          total_reloc[1]+=totalParticles;
          int datasize   = datasizes[idx];
          ASSERT(datasize>0);
          
          Uintah::MPI::Pack(&datasize, 1, MPI_INT, buf, sendsize, &position, pg->getComm());

          int start = position;
          ParticleSubset* pset         = old_dw->getParticleSubset(matl, record->fromPatch);
          ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
          ParticleSubset* send_pset    = record->send_pset;
          posvar->packMPI(buf, sendsize, &position, pg, send_pset, record->toPatch);
          
          for(int v=0;v<numVars;v++){
            ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);
            var->packMPI(buf, sendsize, &position, pg, send_pset);
          }
          int size=position-start;
          if(size < datasize){
            // MPI mis-estimated the size of the message.  For some
            // reason, mpich does this all the time.  We must pad...
            int diff=datasize-size;
            char* junk = scinew char[diff];
            Uintah::MPI::Pack(junk, diff, MPI_CHAR, buf, sendsize, &position, pg->getComm());
            
            ASSERTEQ(position, start+datasize);
            delete[] junk;
          }
          idx++;
        }
      } // matl loop
    }  // patch loop
    ASSERT(position <= sendsize);
    ASSERT(sendsize > 0); 
       
    // Send (isend) the message
    MPI_Request rid;
    int to=iter->first;
    
    DOUT(g_mpi_dbg, "Rank-" << pg->myRank() << " Send relocate msg size " << sendsize << " tag " << RELOCATE_TAG << " to ");

    Uintah::MPI::Isend(buf, sendsize, MPI_PACKED, to, RELOCATE_TAG, pg->getComm(), &rid);

    DOUT(g_mpi_dbg, "Rank-" << " done Sending relocate msg size " << sendsize << " tag " << RELOCATE_TAG << " to " << to);
    
    sendbuffers.push_back(buf);
    sendrequests.push_back(rid);
  }  // scatter records loop

  // Receive, and handle the local case too...
  // Foreach processor, post a receive
  recvbuffers.resize(scatter_records->procs.size());

  // I wish that there was an Iprobe_some call, so that we could do
  // this more dynamically...
  int idx=0;
  for(procmaptype::iterator iter = scatter_records->procs.begin();
                            iter != scatter_records->procs.end(); iter++, idx++){
    if(iter->first == me){
      // Local - put a placeholder here for the buffer and request
      recvbuffers[idx]=0;
      continue;
    }
    
    MPI_Status status;
    Uintah::MPI::Probe(iter->first, RELOCATE_TAG, pg->getComm(), &status);
    //ASSERT(status.MPI_ERROR == 0);      
    
    int size;
    Uintah::MPI::Get_count(&status, MPI_PACKED, &size);
    ASSERT(size != 0);
    
    char* buf = scinew char[size];
    recvbuffers[idx]=buf;

    DOUT(g_mpi_dbg, "Rank-" << pg->myRank() << " Recv relocate msg size " << size << " tag " << RELOCATE_TAG << " from " << iter->first);

    Uintah::MPI::Recv(recvbuffers[idx], size, MPI_PACKED, iter->first, RELOCATE_TAG, pg->getComm(), &status);

    DOUT(g_mpi_dbg, "Rank-" << pg->myRank() << " Done Recving relocate msg size " << size << " tag " << RELOCATE_TAG << " from " << iter->first);

    // Partially unpack
    int position=0;
    int numrecords;
    
    Uintah::MPI::Unpack(buf, size, &position, &numrecords,    1, MPI_INT, pg->getComm());
    
    for(int i=0;i<numrecords;i++){
      int patchid;
      Uintah::MPI::Unpack(buf, size, &position, &patchid,     1, MPI_INT, pg->getComm());

      // find the patch from the id
      const Patch* toPatch = grid->getPatchByID(patchid, coarsestLevel->getIndex());;

      ASSERT(toPatch != 0 && toPatch->getID() == patchid);
      
      int matl;
      Uintah::MPI::Unpack(buf, size, &position, &matl,        1, MPI_INT, pg->getComm());

      int numParticles;
      Uintah::MPI::Unpack(buf, size, &position, &numParticles,1, MPI_INT, pg->getComm());
      
      int datasize;
      Uintah::MPI::Unpack(buf, size, &position, &datasize,    1, MPI_INT, pg->getComm());
      
      char* databuf=buf+position;
      ASSERTEQ( m_lb->getPatchwiseProcessorAssignment(toPatch), me );
      
      scatter_records->saveRecv(toPatch, matl, databuf, datasize, numParticles);
      
      position       += datasize;
      total_reloc[2] += numParticles;
    }
  }
}
//______________________________________________________________________
//
void Relocate::finalizeCommunication()
{
  // Wait to make sure that all of the sends completed
  int numsends = (int)sendrequests.size();
  std::vector<MPI_Status> statii(numsends);
  Uintah::MPI::Waitall(numsends, &sendrequests[0], &statii[0]);

  // delete the buffers
  for(int i=0;i<(int)sendbuffers.size();i++){
    delete[] sendbuffers[i];
  }
  
  for(int i=0;i<(int)recvbuffers.size();i++){
    if(recvbuffers[i]){
      delete[] recvbuffers[i];
    }
  }
  
  sendrequests.clear();
  recvbuffers.clear();
  sendbuffers.clear();
}
//______________________________________________________________________
//
const Patch* findFinePatch(const Uintah::Point& pos, const Patch* guess, Level* fineLevel)
{
  if (guess && guess->containsPointInExtraCells(pos)){
    return guess;
  }
  bool includeExtraCells = false;
  return fineLevel->getPatchFromPoint(pos, includeExtraCells);
}
//______________________________________________________________________
//
const Patch* findCoarsePatch(const Uintah::Point& pos, const Patch* guess, Level* coarseLevel)
{
  if (guess && guess->containsPointInExtraCells(pos)){
    return guess;
  }
  bool includeExtraCells = false;
  return coarseLevel->getPatchFromPoint(pos, includeExtraCells);
}

//______________________________________________________________________
// find all of patches on the current level and the adjacent fine 
// and coarse levels.  The current algorithm finds ALL patches on the adjacent
// fine/coarse level.  Ideally, you only need to return the patches along the
// coarse fine interface.  If this function is ever optimized the patch 
// configurations below are the most challenging.
//
// examples:                                corner case between L-0 P1 & L-1 P2      
//            L-0                               |                    |    
//         +-------+--------+                   |____________________|__  
//         |       |        |                   |      |             |    
//         |   +---+---+    |                   |      |     L-0     |    
//         | 0 |  L-1  | 1  |                   |      |     1       |    
//         |   | 2 | 3 |    |                   |      |             |    
//         |   +---+---+    |                   |------+------+------+-   
//         |       |        |                   |      |      |      |    
//         +-------+--------+                   |  2   |  L-1 |      |    
//                                              |      |      |      |    
//  L-0 patches have neighbors   2 & 3          |______|______|______|
//  L-1 patches have neighbors   0 & 1

void
Relocate::findNeighboringPatches(const Patch* patch,
                                 const Level* level,
                                 const bool findFiner,
                                 const bool findCoarser,
                                 Patch::selectType& AllNeighborPatches)
{
//  cout << "relocate findNeighboringPatch L-"<< level->getIndex() << " patch: " << patch->getID() << " hasFiner: " << hasFiner << " hasCoarser: " << hasCoarser << endl;
  
  // put patch neighbors into a std::set this will automatically
  // delete any duplicate patch entries
  std::set<const Patch*> neighborSet;
  
  //__________________________________
  // current level
  Patch::selectType neighborPatches;

  // Particles are only allowed to be one cell out
  IntVector l = patch->getExtraCellLowIndex()  - IntVector(1,1,1);
  IntVector h = patch->getExtraCellHighIndex() + IntVector(1,1,1);
   
  level->selectPatches(l, h, neighborPatches);
  for(unsigned int i=0; i<neighborPatches.size(); i++){
    const Patch* neighbor=neighborPatches[i];
    neighborSet.insert(neighbor);
  }
  
  //__________________________________
  //  Find coarse level patches
  if(findCoarser){
    IntVector refineRatio = level->getRefinementRatio();
    const Level* coarseLevel = level->getCoarserLevel().get_rep();  

    IntVector fl = patch->getExtraCellLowIndex()  - refineRatio;
    IntVector fh = patch->getExtraCellHighIndex() + refineRatio;
    IntVector cl = level->mapCellToCoarser(fl);     
    IntVector ch = level->mapCellToCoarser(fh);

    Patch::selectType coarsePatches;
    coarseLevel->selectPatches(cl, ch, coarsePatches);

    ASSERT( coarsePatches.size() != 0 );

    for(unsigned int i=0; i<coarsePatches.size(); i++){
      const Patch* neighbor=coarsePatches[i];
      neighborSet.insert(neighbor);
    }
  }
  
  //__________________________________
  // Find the fine level patches.
  if(findFiner){
    const Level* fineLevel = level->getFinerLevel().get_rep();
    
    // Particles are only allowed to be one cell out
    IntVector cl = patch->getExtraCellLowIndex()  - IntVector(1,1,1);
    IntVector ch = patch->getExtraCellHighIndex() + IntVector(1,1,1);
    IntVector fl = level->mapCellToFiner(cl);
    IntVector fh = level->mapCellToFiner(ch);
    
    Patch::selectType finePatches;
    fineLevel->selectPatches(fl, fh,finePatches); 

    for(unsigned int i=0;i<finePatches.size();i++){
      const Patch* finePatch = finePatches[i];
      neighborSet.insert(finePatch);
    }  // fine patches loop
  }

  //__________________________________
  // put the neighborSet into a selectType variable.
  for (std::set<const Patch*>::iterator iter = neighborSet.begin();iter != neighborSet.end();++iter) {
    const Patch* neighbor = *iter;
    AllNeighborPatches.push_back(neighbor);
  }
}
//______________________________________________________________________
//
void
Relocate::relocateParticlesModifies( const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     const Level* coarsestLevelwithParticles )
{
  int total_reloc[3] = {0,0,0};
  if (patches->size() != 0)
  {
    printTask(patches, patches->get(0),coutdbg,"Relocate::relocateParticles");
    int me = pg->myRank();
    
    // First pass: For each of the patches we own, look for particles
    // that left the patch.  Create a scatter record for each one.
    MPIScatterRecords scatter_records;
    int numMatls = (int)reloc_old_labels.size();
    
    Array2<ParticleSubset*> keep_psets(patches->size(), numMatls);
    keep_psets.initialize(0);
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      const Level* level = patch->getLevel();
      
      // AMR
      const Level* curLevel = patch->getLevel();
      bool findFiner   = curLevel->hasFinerLevel();
      bool findCoarser = curLevel->hasCoarserLevel() && curLevel->getIndex() > coarsestLevelwithParticles->getIndex();
      Level* fineLevel=0;
      Level* coarseLevel=0;
      
      if( findFiner ) {
        fineLevel = (Level*) curLevel->getFinerLevel().get_rep();
      }
      if( findCoarser ) {
        coarseLevel = (Level*) curLevel->getCoarserLevel().get_rep();
      }
      
      Patch::selectType neighborPatches;
      findNeighboringPatches(patch, level, findFiner, findCoarser, neighborPatches);
      
      // Find all of the neighborPatches, and add them to a set
      for(unsigned int i=0; i<neighborPatches.size(); i++){
        const Patch* neighbor=neighborPatches[i];
        scatter_records.addNeighbor( m_lb, pg, neighbor );
      }
      
      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
        unsigned int numParticles     = pset->numParticles();
        
        constParticleVariable<Point> px;
        new_dw->get(px, reloc_old_posLabel, pset);
        
        ParticleSubset* keep_pset    = scinew ParticleSubset(0, -1, 0);
        ParticleSubset* delete_pset  = new_dw->getDeleteSubset(matl, patch);
        
        keep_pset->expand(numParticles);
        
        
        // Look for particles that left the patch,
        // and if they are not in the delete set, put them in relocset
        
        ParticleSubset::iterator delete_iter = delete_pset->begin();
        
        ASSERT( std::is_sorted(pset->begin(), pset->end()) );
        ASSERT( std::is_sorted(delete_pset->begin(), delete_pset->end()) );
        ASSERT( pset->begin() == pset->end() || *pset->begin() == 0 );
        
        // The previous Particle's relocation patch
        const Patch* PP_ToPatch_FL = 0;   // on fine level
        const Patch* PP_ToPatch_CL = 0;   // on coarse level
        const Patch* PP_ToPatch    = 0;
        
        
        for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
          particleIndex idx = *iter;
          
          const Patch* toPatch = 0; // patch to relocate particles to
          
          //__________________________________
          // Does this particle belong to the delete particle set?
          // The delete particle set is computed in MPM
          if (delete_iter != delete_pset->end() && idx == *delete_iter) {
            // all you need to do to delete a particle is neither keep it or
            // relocate it.  So just go to the next deleted particle and wait for a match
            delete_iter++;
          }
          
          //__________________________________
          //  Has particle moved to a finer level?
          else if (fineLevel && (toPatch = findFinePatch(px[idx], PP_ToPatch_FL, fineLevel) ) ) {
            PP_ToPatch_FL = toPatch;
          }
          
          //__________________________________
          //  Does this patch contains this particle?
          else if(patch->containsPoint(px[idx])){
            // is particle going to a finer patch?  Note, a particle does not have to leave the current patch
            // to go to a finer patch
            keep_pset->addParticle(idx);
          }
          
          //__________________________________
          //Particle is not on the current patch find where it went
          else {
            
            //__________________________________
            //  Did the particle move to the same patch as the previous particle?
            //  (optimization)
            if (PP_ToPatch && PP_ToPatch->containsPointInExtraCells(px[idx])){
              toPatch = PP_ToPatch;
            }else {
              //__________________________________
              //  Search for the new patch that the particle belongs to on this level.
              bool includeExtraCells = false;
              toPatch = level->getPatchFromPoint(px[idx], includeExtraCells);
              PP_ToPatch = toPatch;
              
              //__________________________________
              // The particle is not in the surrounding patches
              // has it moved to a coarser level?
              if (toPatch == 0 && coarseLevel){
                toPatch = findCoarsePatch(px[idx], PP_ToPatch_CL, coarseLevel);
                
                PP_ToPatch_CL = toPatch;
#if SCI_ASSERTION_LEVEL >= 1
                if(!toPatch && level->containsPoint(px[idx])){
                  // Make sure that the particle really left the world
                  static ProgressiveWarning warn("A particle just travelled from one patch to another non-adjacent patch.  It has been deleted and we're moving on.",10);
                  warn.invoke();
                }
#endif
              }
            }  // search for new patch that particle belongs to
          }  // not on current patch
          
          //__________________________________
          // We know which patch the particle is
          // going to be moved to, add it to a scatter record
          if (toPatch) {
            total_reloc[0]++;
            int toLevelIndex = toPatch->getLevel()->getIndex();
            ScatterRecord* record = scatter_records.findOrInsertRecord(patch, toPatch, matl, toLevelIndex, pset);
            record->send_pset->addParticle(idx);
          }
        }  // pset loop
        
        //__________________________________
        //  No particles have left the patch
        if(keep_pset->numParticles() == numParticles){
          delete keep_pset;
          keep_pset=pset;
        }
        
        keep_pset->addReference();
        keep_psets(p, m)=keep_pset;
      } // matls loop
    }  // patches loop
    
    //__________________________________
    if (pg->nRanks() > 1) {
      // send the particles where they need to go
      exchangeParticles(pg, patches, matls, old_dw, new_dw, &scatter_records, total_reloc);
    }
    
    //__________________________________
    // Now go through each of our patches, and do the merge.  Also handle the local case
    for(int p=0;p<patches->size();p++){
      const Patch* toPatch = patches->get(p);
      const Level* level   = toPatch->getLevel();
      
      // AMR related
      const Level* curLevel = toPatch->getLevel();
      int curLevelIndex     = curLevel->getIndex();
      bool findFiner   = curLevel->hasFinerLevel();
      bool findCoarser = curLevel->hasCoarserLevel() && curLevel->getIndex() > coarsestLevelwithParticles->getIndex();
      
      Patch::selectType neighborPatches;
      findNeighboringPatches(toPatch, level, findFiner, findCoarser, neighborPatches);
      
      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        
        int numVars = (int)reloc_old_labels[m].size();
        std::vector<const Patch*> fromPatches;
        std::vector<ParticleSubset*> subsets;
        
        ParticleSubset* keep_pset = keep_psets(p, m);
        ASSERT( keep_pset != 0 );
        
        fromPatches.push_back(toPatch);
        subsets.push_back(keep_pset);
        
        // loop over all neighboring patches and find all of the 'from' patches
        // on this processor
        for(int i=0;i<(int)neighborPatches.size();i++){
          const Patch* fromPatch=neighborPatches[i];
          
          int fromProc = m_lb->getPatchwiseProcessorAssignment(fromPatch->getRealPatch());
          ASSERTRANGE( fromProc, 0, pg->nRanks() );
          
          if(fromProc == me){
            ScatterRecord* record = scatter_records.findRecord(fromPatch, toPatch, matl, curLevelIndex);
            if(record){
              fromPatches.push_back(fromPatch);
              subsets.push_back(record->send_pset);
            }
          } // fromProc==me
        }  // neighbor patches
        
        MPIRecvBuffer* recvs = scatter_records.findRecv(toPatch, matl);
        
        // create a map for the new particles
        std::map<const VarLabel*, ParticleVariableBase*>* newParticles_map = 0;
        newParticles_map = new_dw->getNewParticleState(matl, toPatch);
        bool adding_new_particles = false;
        
        if (newParticles_map){
          adding_new_particles = true;
        }
        
        ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, toPatch);
        
        //__________________________________
        // Particles haven't moved, carry the old data forward
        if(recvs == 0 && subsets.size() == 1 && keep_pset == orig_pset && !adding_new_particles){
          // carry forward old data
          new_dw->saveParticleSubset(orig_pset, matl, toPatch);
          
          // particle position
          ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
          new_dw->put(*posvar, reloc_new_posLabel);
          
          // all other variables
          for(int v=0;v<numVars;v++){
            ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], orig_pset);
            new_dw->put(*var, reloc_new_labels[m][v]);
          }
        } else {
          
          //__________________________________
          // Particles have moved
          int numOldVariables = (int)subsets.size();
          
#if 0
          if(newParticles_map){
            
            // bulletproofing
            map<const VarLabel*, ParticleVariableBase*>::iterator piter;
            piter = newParticles_map->find(reloc_new_posLabel);
            
            if(piter == newParticles_map->end()){
              throw InternalError("didnt create new position", __FILE__, __LINE__);
            }
            
            ParticleVariableBase* addedPos = piter->second;
            subsets.push_back(addedPos->getParticleSubset());
          }
#endif
          
          int totalParticles=0;
          for(int i=0;i<(int)subsets.size();i++){
            totalParticles+=subsets[i]->numParticles();
          }
          
          int numRemote=0;
          for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
            numRemote+=buf->numParticles;
          }
          totalParticles+=numRemote;
          
          ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, matl, toPatch);
          
          //__________________________________
          // particle position
          // Merge local portion
          std::vector<ParticleVariableBase*> invars(subsets.size());
          for(int i=0;i<(int)numOldVariables;i++){
            invars[i]=new_dw->getParticleVariable(reloc_old_posLabel, matl, fromPatches[i]);
          }
          
          if(newParticles_map){
            // bulletproofing
            std::map<const VarLabel*, ParticleVariableBase*>::iterator piter;
            piter = newParticles_map->find(reloc_new_posLabel);
            
            if(piter == newParticles_map->end()){
              throw InternalError("didnt create new position", __FILE__, __LINE__);
            }
            
            
            ParticleVariableBase* addedPos = piter->second;
            invars[subsets.size()-1] = addedPos;
            fromPatches.push_back(toPatch);
          }
          
          // particle position
          ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
          ParticleVariableBase* newpos = posvar->clone();
          newpos->gather(newsubset, subsets, invars, fromPatches, numRemote);
          
          //__________________________________
          // other particle variables
          std::vector<ParticleVariableBase*> vars(numVars);
          
          for(int v=0;v<numVars;v++){
            const VarLabel* label = reloc_old_labels[m][v];
            ParticleVariableBase* var = new_dw->getParticleVariable(label, orig_pset);
            
            for(int i=0;i<numOldVariables;i++){
              invars[i]=new_dw->getParticleVariable(label, matl, fromPatches[i]);
            }
            
            if(newParticles_map){
              // bulletproofing
              std::map<const VarLabel*, ParticleVariableBase*>::iterator piter;
              piter = newParticles_map->find(reloc_new_labels[m][v]);
              
              if(piter == newParticles_map->end()) {
                throw InternalError("didnt create new variable of this type", __FILE__, __LINE__);
              }
              
              ParticleVariableBase* addedVar = piter->second;
              invars[subsets.size()-1] = addedVar;
            }
            
            ParticleVariableBase* newvar = var->clone();
            newvar->gather(newsubset, subsets, invars, fromPatches, numRemote);
            vars[v]=newvar;
          }  // numVars
          
          //__________________________________
          // Unpack MPI portion
          particleIndex idx = totalParticles-numRemote;
          for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
            int position=0;
            ParticleSubset* unpackset = scinew ParticleSubset(0, matl, toPatch);
            unpackset->resize(buf->numParticles);
            
            for(int p=0;p<buf->numParticles;p++,idx++){
              unpackset->set(p, idx);
            }
            
            newpos->unpackMPI(buf->databuf, buf->bufsize, &position, pg, unpackset);
            for(int v=0;v<numVars;v++){
              vars[v]->unpackMPI(buf->databuf, buf->bufsize, &position,pg, unpackset);
            }
            
            ASSERT( position <= buf->bufsize );
            delete unpackset;
          }  // MPI portion
          
          ASSERTEQ( idx, totalParticles );
          
#if 0
          for(int v=0;v<numVars;v++){
            const VarLabel* label = reloc_new_labels[m][v];
            if (label == particleIDLabel_)
              break;
          }
          
          // must have a p.particleID variable in reloc labels
          ASSERT( v < numVars );
          newsubset->sort(vars[v] /* particleID variable */);
#endif
          
          // Put the data back in the data warehouse
          new_dw->put(*newpos, reloc_new_posLabel);
          
          delete newpos;
          
          for(int v=0;v<numVars;v++){
            new_dw->put(*vars[v], reloc_new_labels[m][v]);
            delete vars[v];
          }
        }  // particles have moved
        if(keep_pset->removeReference()){
          delete keep_pset;
        }
      }  // matls loop
    }  // patches loop
    
    DOUT(g_total_reloc, "total_reloc: " << total_reloc[0] << ", " << total_reloc[1] << ", " << total_reloc[2]);

  }  // patch size !-= 0
  
  if (pg->nRanks() > 1){
    finalizeCommunication();
  }
  
  // Finally, copy the relocated variables back into the original variables :)
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    for (int m = 0; m < matls->size(); m++) {
      const int matl = matls->get(m);
      ParticleVariableBase* pPos = new_dw->getParticleVariable(reloc_new_posLabel, matl, patch);
      new_dw->put(*pPos, reloc_old_posLabel, true);

      // go over the list of particle variables
      for (size_t i = 0; i < reloc_new_labels[m].size(); i++) {
        const VarLabel* pVarLabel = reloc_old_labels[m][i];
        const VarLabel* relocVarLabel = reloc_new_labels[m][i];
        ParticleVariableBase* pvar = new_dw->getParticleVariable(relocVarLabel,matl, patch);
        new_dw->put(*pvar, pVarLabel,true);
      }
    }
  }
  
}
//______________________________________________________________________
//
void
Relocate::relocateParticles(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const Level* coarsestLevelwithParticles)
{
  int total_reloc[3] = {0,0,0};
  if (patches->size() != 0) {
    printTask(patches, patches->get(0),coutdbg,"Relocate::relocateParticles");
    int me = pg->myRank();

    // First pass: For each of the patches we own, look for particles
    // that left the patch.  Create a scatter record for each of the patches
    MPIScatterRecords scatter_records;
    int numMatls = (int)reloc_old_labels.size();
    
    Array2<ParticleSubset*> keep_psets(patches->size(), numMatls);
    keep_psets.initialize(0);
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      const Level* level = patch->getLevel();

      // AMR
      const Level* curLevel = patch->getLevel();
      bool findFiner   = curLevel->hasFinerLevel();
      bool findCoarser = curLevel->hasCoarserLevel() && curLevel->getIndex() > coarsestLevelwithParticles->getIndex();
      Level* fineLevel=0;
      Level* coarseLevel=0;

      if(findFiner){
        fineLevel = (Level*) curLevel->getFinerLevel().get_rep();
      }
      if(findCoarser){
        coarseLevel = (Level*) curLevel->getCoarserLevel().get_rep();
      }
      
      Patch::selectType neighborPatches;
      findNeighboringPatches(patch, level, findFiner, 
                                           findCoarser, neighborPatches);

      // Find all of the neighborPatches, and add them to a set
      for(unsigned int i=0; i<neighborPatches.size(); i++){
        const Patch* neighbor=neighborPatches[i];
        scatter_records.addNeighbor( m_lb, pg, neighbor );
      }

      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
        unsigned int numParticles     = pset->numParticles();

        constParticleVariable<Point> px;
        new_dw->get(px, reloc_old_posLabel, pset);

        ParticleSubset* keep_pset    = scinew ParticleSubset(0, -1, 0);
        ParticleSubset* delete_pset  = new_dw->getDeleteSubset(matl, patch);

        keep_pset->expand(numParticles);

        // Look for particles that left the patch, 
        // and if they are not in the delete set, put them in relocset

        ParticleSubset::iterator delete_iter = delete_pset->begin();

        ASSERT(std::is_sorted(pset->begin(), pset->end()));
        ASSERT(std::is_sorted(delete_pset->begin(), delete_pset->end()));
        ASSERT(pset->begin() == pset->end() || *pset->begin() == 0);

        // The previous Particle's relocation patch
        const Patch* PP_ToPatch_FL = 0;   // on fine level
        const Patch* PP_ToPatch_CL = 0;   // on coarse level
        const Patch* PP_ToPatch    = 0;

        for(ParticleSubset::iterator iter  = pset->begin(); 
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          
          const Patch* toPatch = 0; // patch to relocate particles to

          //__________________________________
          // Does this particle belong to the delete particle set?
          // The delete particle set is computed in MPM
          if (delete_iter != delete_pset->end() && idx == *delete_iter) {
            // all you need to do to delete a particle is neither keep it or 
            // relocate it.  So just go to the next deleted particle
            // and wait for a match
            delete_iter++;
          }

          //__________________________________
          //  Has particle moved to a finer level?
          else if (fineLevel && (toPatch = findFinePatch(px[idx], PP_ToPatch_FL, fineLevel) ) ) {
            PP_ToPatch_FL = toPatch;
          } 

          //__________________________________
          //  Does this patch contains this particle?
          else if(patch->containsPoint(px[idx])){
            // is particle going to a finer patch?  Note, a particle does not have to leave the current patch
            // to go to a finer patch
            keep_pset->addParticle(idx);
          }
          
          //__________________________________
          //Particle is not on the current patch find where it went
          else {
            //__________________________________
            //  Did the particle move to the same patch as the previous particle?
            //  (optimization)
            if (PP_ToPatch && PP_ToPatch->containsPointInExtraCells(px[idx])){
              toPatch = PP_ToPatch;
            }else { 
              //__________________________________
              //  Search for the new patch that the particle belongs to on this level.
              bool includeExtraCells = false;
              toPatch = level->getPatchFromPoint(px[idx], includeExtraCells);
              PP_ToPatch = toPatch;
              
              //__________________________________
              // The particle is not in the surrounding patches
              // has it moved to a coarser level?
              if (toPatch == 0 && coarseLevel){
                  toPatch = findCoarsePatch(px[idx], PP_ToPatch_CL, coarseLevel);

                  PP_ToPatch_CL = toPatch;
#if SCI_ASSERTION_LEVEL >= 1
                if(!toPatch && level->containsPoint(px[idx])){
                  // Make sure that the particle really left the world
                  static ProgressiveWarning warn("A particle just travelled from one patch to another non-adjacent patch.  It has been deleted and we're moving on.",10);
                  warn.invoke();
                }
#endif
              }
            }  // search for new patch that particle belongs to
          }  // not on current patch
          
          //__________________________________
          // We know which patch the particle is
          // going to be moved to, add it to a scatter record
          if (toPatch) {
            total_reloc[0]++;
            int toLevelIndex = toPatch->getLevel()->getIndex();
            ScatterRecord* record = scatter_records.findOrInsertRecord(patch,toPatch,matl,toLevelIndex,pset);
            record->send_pset->addParticle(idx);
          }
        }  // pset loop
        
        //__________________________________
        //  No particles have left the patch
        if(keep_pset->numParticles() == numParticles){
          delete keep_pset;
          keep_pset=pset;
        }
        
        keep_pset->addReference();
        keep_psets(p, m)=keep_pset;
      } // matls loop
    }  // patches loop

    //__________________________________
    if (pg->nRanks() > 1) {
      // send the particles where they need to go
      exchangeParticles(pg, patches, matls, old_dw, new_dw,
                                     &scatter_records, total_reloc);
    }

    //__________________________________
    // Now go through each of our patches, and do the merge.
    // Also handle the local case
    for(int p=0;p<patches->size();p++){
      const Patch* toPatch = patches->get(p);
      const Level* level   = toPatch->getLevel();

      // AMR related
      const Level* curLevel = toPatch->getLevel();
      int curLevelIndex     = curLevel->getIndex();
      bool findFiner   = curLevel->hasFinerLevel();
      bool findCoarser = curLevel->hasCoarserLevel() &&
                  curLevel->getIndex() > coarsestLevelwithParticles->getIndex();
      
      Patch::selectType neighborPatches;
      findNeighboringPatches(toPatch, level, findFiner,
                                             findCoarser, neighborPatches);

      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        
        int numVars = (int)reloc_old_labels[m].size();
        std::vector<const Patch*> fromPatches;
        std::vector<ParticleSubset*> subsets;
        
        ParticleSubset* keep_pset = keep_psets(p, m);
        ASSERT(keep_pset != 0);
        
        fromPatches.push_back(toPatch);
        subsets.push_back(keep_pset);

        // loop over all neighboring patches and find all of the 'from' patches
        // on this processor
        for(int i=0;i<(int)neighborPatches.size();i++){
          const Patch* fromPatch=neighborPatches[i];

          int fromProc = m_lb->getPatchwiseProcessorAssignment(fromPatch->getRealPatch());
          ASSERTRANGE(fromProc, 0, pg->nRanks());
          
          if(fromProc == me){
            ScatterRecord* record = scatter_records.findRecord(fromPatch,
                                                  toPatch, matl, curLevelIndex);
            if(record){
              fromPatches.push_back(fromPatch);
              subsets.push_back(record->send_pset);
            }
          } // fromProc==me
        }  // neighbor patches

        MPIRecvBuffer* recvs = scatter_records.findRecv(toPatch, matl);

        // JG:  Potential weed left over from when particles were created
        // midrun in MPM?
        bool adding_new_particles = false;
#if 0
        // create a map for the new particles
        map<const VarLabel*, ParticleVariableBase*>* newParticles_map = 0;
        newParticles_map = new_dw->getNewParticleState(matl, toPatch);
        bool adding_new_particles = false;
        
        if (newParticles_map){
          adding_new_particles = true;
        }
#endif
        
        ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, toPatch);

        //__________________________________
        // Particles haven't moved, carry the old data forward
        if(recvs == 0 && subsets.size() == 1 && 
           keep_pset == orig_pset && !adding_new_particles){
          // carry forward old data
          new_dw->saveParticleSubset(orig_pset, matl, toPatch);
          
          // particle position
          ParticleVariableBase* posvar =
                     new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
          new_dw->put(*posvar, reloc_new_posLabel);
          
          // all other variables
          for(int v=0;v<numVars;v++){
            ParticleVariableBase* var =
                 new_dw->getParticleVariable(reloc_old_labels[m][v], orig_pset);
            new_dw->put(*var, reloc_new_labels[m][v]);
          }
        } else {

          // Particles have moved
          // carry forward old data
          int numOldVariables = (int)subsets.size();
          
          // JG:  Potential weed left over from when particles were created
          // midrun in MPM?
#if 0
          if(newParticles_map){
            // bulletproofing
            map<const VarLabel*, ParticleVariableBase*>::iterator piter;
            piter = newParticles_map->find(reloc_new_posLabel);
            
            if(piter == newParticles_map->end()){
              throw InternalError("didnt create new position", 
                                                           __FILE__, __LINE__);
            }

            ParticleVariableBase* addedPos = piter->second;
            subsets.push_back(addedPos->getParticleSubset());
          }
#endif

          int totalParticles=0;
          for(int i=0;i<(int)subsets.size();i++){
            totalParticles+=subsets[i]->numParticles();
          }
          
          int numRemote=0;
          for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
            numRemote+=buf->numParticles;
          }
          totalParticles+=numRemote;

          ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, matl, toPatch);

          //__________________________________
          // particle position
          // Merge local portion
          std::vector<ParticleVariableBase*> invars(subsets.size());
          for(int i=0;i<(int)numOldVariables;i++){
            invars[i]=new_dw->getParticleVariable(reloc_old_posLabel, matl, fromPatches[i]);
          }

          // JG:  Potential weed left over from when particles were created
          // midrun in MPM?
#if 0
          if(newParticles_map){
            // bulletproofing
            map<const VarLabel*, ParticleVariableBase*>::iterator piter;
            piter = newParticles_map->find(reloc_new_posLabel);
            
            if(piter == newParticles_map->end()){
              throw InternalError("didnt create new position",
                                                           __FILE__, __LINE__);
            }
            ParticleVariableBase* addedPos = piter->second;
            invars[subsets.size()-1] = addedPos;
            fromPatches.push_back(toPatch);
          }
#endif

          // particle position
          ParticleVariableBase* posvar = 
                     new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
          ParticleVariableBase* newpos = posvar->clone();
          newpos->gather(newsubset, subsets, invars, fromPatches, numRemote);

          //__________________________________
          // other particle variables
          std::vector<ParticleVariableBase*> vars(numVars);
          
          for(int v=0;v<numVars;v++){
            const VarLabel* label = reloc_old_labels[m][v];
            ParticleVariableBase* var =
                                  new_dw->getParticleVariable(label, orig_pset);
            for(int i=0;i<numOldVariables;i++){
              invars[i]=new_dw->getParticleVariable(label,matl,fromPatches[i]);
            }

#if 0
            if(newParticles_map){
              // bulletproofing
              map<const VarLabel*, ParticleVariableBase*>::iterator piter;
              piter = newParticles_map->find(reloc_new_labels[m][v]);
              
              if(piter == newParticles_map->end()) {
                throw InternalError("didnt create new variable of this type", __FILE__, __LINE__);
              }
              
              ParticleVariableBase* addedVar = piter->second;
              invars[subsets.size()-1] = addedVar;
            }
#endif

            ParticleVariableBase* newvar = var->clone();
            newvar->gather(newsubset, subsets, invars, fromPatches, numRemote);
            vars[v]=newvar;
          }  // numVars

          //__________________________________
          // Unpack MPI portion
          particleIndex idx = totalParticles-numRemote;
          for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
            int position=0;
            ParticleSubset* unpackset = scinew ParticleSubset(0, matl, toPatch);
            unpackset->resize(buf->numParticles);
            
            for(int p=0;p<buf->numParticles;p++,idx++){
              unpackset->set(p, idx);
            }
            
            newpos->unpackMPI(buf->databuf, buf->bufsize, &position, pg, unpackset);
            for(int v=0;v<numVars;v++){
              vars[v]->unpackMPI(buf->databuf, buf->bufsize, &position,pg, unpackset);
            }
            
            ASSERT(position <= buf->bufsize);
            delete unpackset;
          }  // MPI portion
          
          ASSERTEQ(idx, totalParticles);

#if 0
          for(int v=0;v<numVars;v++){
            const VarLabel* label = reloc_new_labels[m][v];
            if (label == particleIDLabel_)
              break;
          }

          // must have a p.particleID variable in reloc labels
          ASSERT(v < numVars); 
          newsubset->sort(vars[v] /* particleID variable */);
#endif
  
          // Put the data back in the data warehouse
          new_dw->put(*newpos, reloc_new_posLabel);

          delete newpos;
          
          for(int v=0;v<numVars;v++){
            new_dw->put(*vars[v], reloc_new_labels[m][v]);
            delete vars[v];
          }
        }  // particles have moved 
        if(keep_pset->removeReference()){
          delete keep_pset;
        }
      }  // matls loop
    }  // patches loop

    DOUT(g_total_reloc, "total_reloc: " << total_reloc[0] << ", " << total_reloc[1] << ", " << total_reloc[2]);

  }  // patch size !-= 0
  

  if (pg->nRanks() > 1){
    finalizeCommunication();
  }

} // end relocateParticles()
