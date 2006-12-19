
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Util/ProgressiveWarning.h>

#include <Core/Containers/Array2.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>

#include <map>
#include <set>

#define RELOCATE_TAG            0x3fff

using namespace std;
using namespace Uintah;

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCISHARE SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;
extern DebugStream mpidbg;

Relocate::Relocate()
{
  reloc_old_posLabel = reloc_new_posLabel = 0;
  reloc_matls = 0;
}

Relocate::~Relocate()
{
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
}

namespace Uintah {
  struct ScatterRecord {
    const Patch* fromPatch;
    const Patch* toPatch;    
    IntVector vectorToNeighbor;
    int matl;    
    ParticleSubset* sendset;
    
    ScatterRecord(const Patch* fromPatch, const Patch* toPatch, int matl)
      : fromPatch(fromPatch), toPatch(toPatch), matl(matl), sendset(0)
    {
      ASSERT(fromPatch != 0);
      ASSERT(toPatch != 0);
      vectorToNeighbor = toPatch->getLowIndex() - fromPatch->getLowIndex();
    }

    // Note that when the ScatterRecord going from a real patch to
    // a virtual patch has an equivalent representation going from
    // a virtual patch to a real patch (wrap-around, periodic bound. cond.).
    bool equivalent(const ScatterRecord& sr)
    { return (toPatch->getRealPatch() == sr.toPatch->getRealPatch()) &&
	(matl == sr.matl) && (vectorToNeighbor == sr.vectorToNeighbor);
    }
  };

  typedef multimap<pair<const Patch*, int>, ScatterRecord*> maptype;


  struct CompareScatterRecord {
    bool operator()(const ScatterRecord* sr1, const ScatterRecord* sr2) const
    {
      return
	((sr1->toPatch->getRealPatch() != sr2->toPatch->getRealPatch()) ?
	 (sr1->toPatch->getRealPatch() < sr2->toPatch->getRealPatch()) :
	 ((sr1->matl != sr2->matl) ? (sr1->matl < sr2->matl) :
	  compareIntVector(sr1->vectorToNeighbor, sr2->vectorToNeighbor)));
    }
    bool compareIntVector(const IntVector& v1, const IntVector& v2) const
    {
      return (v1.x() != v2.x()) ? (v1.x() < v2.x()) :
	((v1.y() != v2.y()) ? (v1.y() < v2.y()) : (v1.z() < v2.z()));
    }    
  };

  typedef vector<const Patch*> patchestype;

  struct MPIScatterProcessorRecord {
    patchestype patches;
    void sortPatches();
  };

  typedef map<int, MPIScatterProcessorRecord*> procmaptype;


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
  typedef map<pair<const Patch*, int>, MPIRecvBuffer*> recvmaptype;

  class MPIScatterRecords {
  public:
    // map the to patch and matl to the ScatterRecord
    maptype records;
    
    procmaptype procs;

    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl,
			      ParticleSubset* pset);
    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl);
    void addNeighbor(LoadBalancer* lb, const ProcessorGroup* pg,
		     const Patch* to);

    recvmaptype recvs;
    void saveRecv(const Patch* to, int matl,
		  char* databuf, int bufsize, int numParticles);
    MPIRecvBuffer* findRecv(const Patch* to, int matl);

    ~MPIScatterRecords();
  };
} // End namespace Uintah


void
Relocate::scheduleParticleRelocation(Scheduler* sched,
				     const ProcessorGroup* pg,
				     LoadBalancer* lb,
				     const LevelP& level,
				     const VarLabel* old_posLabel,
				     const vector<vector<const VarLabel*> >& old_labels,
				     const VarLabel* new_posLabel,
				     const vector<vector<const VarLabel*> >& new_labels,
				     const VarLabel* particleIDLabel,
				     const MaterialSet* matls)
{
  // Only allow particles at the finest level for now
//  if(level->getIndex() != level->getGrid()->numLevels()-1)
//    return;
  reloc_old_posLabel = old_posLabel;
  reloc_old_labels = old_labels;
  reloc_new_posLabel = new_posLabel;
  reloc_new_labels = new_labels;
  particleIDLabel_ = particleIDLabel;
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
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
  for (int m = 0; m< numMatls; m++)
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());
  Task* t = scinew Task("Relocate::relocateParticles",
			this, &Relocate::relocateParticles);
  if(lb)
    t->usesMPI();
  t->requires( Task::NewDW, old_posLabel, Ghost::None);
  for(int m=0;m < numMatls;m++){
    MaterialSubset* thismatl = scinew MaterialSubset();
    thismatl->add(matlsub->get(m));
    for(int i=0;i<(int)old_labels[m].size();i++)
      t->requires( Task::NewDW, old_labels[m][i], thismatl, Ghost::None);

    t->computes( new_posLabel, thismatl);
    for(int i=0;i<(int)new_labels[m].size();i++)
      t->computes(new_labels[m][i], thismatl);
  }
  PatchSet* patches;
  if(!level->hasFinerLevel())
    // only case since the below version isn't const
    patches = const_cast<PatchSet*>(lb->getPerProcessorPatchSet(level)); 
  else {
    GridP grid = level->getGrid();
    // make per-proc patch set of each level >= level
    patches = scinew PatchSet();
    patches->createEmptySubsets(pg->size());
    for (int i = level->getIndex(); i < grid->numLevels(); i++) {
      const PatchSet* p = lb->getPerProcessorPatchSet(grid->getLevel(i));
      for (int proc = 0; proc < pg->size(); proc++) {
        for (int j = 0; j < p->getSubset(proc)->size(); i++) {
          const Patch* patch = p->getSubset(proc)->get(j);
          patches->getSubset(lb->getPatchwiseProcessorAssignment(patch))->add(patch);
        }
      }
    }
  }
  t->setType(Task::OncePerProc);
  sched->addTask(t, patches, matls);
  this->lb=lb;
}

void MPIScatterRecords::saveRecv(const Patch* to, int matl,
				 char* databuf, int datasize, int numParticles)
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

MPIRecvBuffer* MPIScatterRecords::findRecv(const Patch* to, int matl)
{
  recvmaptype::iterator iter = recvs.find(make_pair(to, matl));
  if(iter == recvs.end())
    return 0;
  else
    return iter->second;
}

ScatterRecord* MPIScatterRecords::findRecord(const Patch* from,
					     const Patch* to, int matl,
					     ParticleSubset* pset)
{
  ASSERT(to != 0);
  IntVector vectorToNeighbor = to->getLowIndex() - from->getLowIndex();
  const Patch* realTo = to->getRealPatch();

  pair<maptype::iterator, maptype::iterator> pr =
    records.equal_range(make_pair(realTo, matl));
  for(;pr.first != pr.second;pr.first++){
    if(pr.first->second->vectorToNeighbor == vectorToNeighbor)
      break;
  }
  if(pr.first == pr.second){
    ScatterRecord* rec = scinew ScatterRecord(from, to, matl);
    rec->sendset = scinew ParticleSubset(pset->getParticleSet(), false, -1, 0, 0);
    records.insert(maptype::value_type(make_pair(realTo, matl), rec));
    return rec;
  } else {
    ASSERT(pr.first->second->equivalent(ScatterRecord(from, to, matl)));
    return pr.first->second;
  }
}

ScatterRecord* MPIScatterRecords::findRecord(const Patch* from,
					     const Patch* to, int matl)
{
  ASSERT(to != 0);
  IntVector vectorToNeighbor = to->getLowIndex() - from->getLowIndex();
  const Patch* realTo = to->getRealPatch();

  pair<maptype::iterator, maptype::iterator> pr =
    records.equal_range(make_pair(realTo, matl));
  for(;pr.first != pr.second;pr.first++){
    if(pr.first->second->vectorToNeighbor == vectorToNeighbor)
      break;
  }
  if(pr.first == pr.second){
    return 0;
  } else {
    return pr.first->second;
  }
}

static bool ComparePatches(const Patch* p1, const Patch* p2)
{
  return p1->getID() < p2->getID();
}

void MPIScatterProcessorRecord::sortPatches()
{
  sort(patches.begin(), patches.end(), ComparePatches);
}

void MPIScatterRecords::addNeighbor(LoadBalancer* lb, const ProcessorGroup* pg,
				    const Patch* neighbor)
{
  neighbor = neighbor->getRealPatch();
  int toProc = lb->getPatchwiseProcessorAssignment(neighbor);
  ASSERTRANGE(toProc, 0, pg->size());
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
    for(i=0;i<(int)pr->patches.size();i++)
      if(pr->patches[i] == neighbor)
	break;
    if(i==(int)pr->patches.size())
      pr->patches.push_back(neighbor);
  }
}

MPIScatterRecords::~MPIScatterRecords()
{
  for(procmaptype::iterator iter = procs.begin(); iter != procs.end(); iter++)
    delete iter->second;
  for(maptype::iterator mapiter = records.begin(); mapiter != records.end();
      mapiter++){
    delete mapiter->second->sendset;    
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

void
Relocate::exchangeParticles(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw, MPIScatterRecords* scatter_records,
                            int total_reloc[3])
{
  // this level is the coarsest level involved in the relocation
  const Level* coarsestLevel = patches->get(0)->getLevel();
  GridP grid = coarsestLevel->getGrid();

  int numMatls = (int)reloc_old_labels.size();

  int me = pg->myrank();
  for(procmaptype::iterator iter = scatter_records->procs.begin();
      iter != scatter_records->procs.end(); iter++){
    if(iter->first == me){
      // Local
      continue;
    }
    MPIScatterProcessorRecord* pr = iter->second;
    pr->sortPatches();

    // Go through once to calc the size of the message
    int psize;
    MPI_Pack_size(1, MPI_INT, pg->getComm(), &psize);
    int sendsize=psize; // One for the count of active patches
    int numactive=0;
    vector<int> datasizes;
    for(patchestype::iterator it = pr->patches.begin();
	it != pr->patches.end(); it++){
      const Patch* toPatch = *it;
      for(int m=0;m<numMatls;m++){
        int matl = matls->get(m);
	int numVars = (int)reloc_old_labels[m].size();
	int numParticles=0;
	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records->records.equal_range(make_pair(toPatch, matl));
	for(;pr.first != pr.second; pr.first++){
	  numactive++;
	  int psize;
	  MPI_Pack_size(4, MPI_INT, pg->getComm(), &psize);
	  sendsize += psize; // Patch ID, matl #, # particles, datasize
	  int orig_sendsize=sendsize;
	  ScatterRecord* record = pr.first->second;
	  int np = record->sendset->numParticles();
	  numParticles += np;
	  ParticleSubset* pset = old_dw->getParticleSubset(matl, record->fromPatch);
	  ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
	  ParticleSubset* sendset=record->sendset;
	  posvar->packsizeMPI(&sendsize, pg, sendset);
	  for(int v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);
	    var->packsizeMPI(&sendsize, pg, sendset);
	  }
	  int datasize=sendsize-orig_sendsize;
	  datasizes.push_back(datasize);
	}
      }
    }
    // Create the buffer for this message
    char* buf = scinew char[sendsize];
    int position=0;

    // And go through it again to pack the message
    int idx=0;
    MPI_Pack(&numactive, 1, MPI_INT, buf, sendsize, &position, pg->getComm());
    for(patchestype::iterator it = pr->patches.begin();
	it != pr->patches.end(); it++){
      const Patch* toPatch = *it;
      for(int m=0;m<numMatls;m++){
        int matl = matls->get(m);
	int numVars = (int)reloc_old_labels[m].size();

	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records->records.equal_range(make_pair(toPatch, matl));
	for(;pr.first != pr.second; pr.first++){
	  int patchid = toPatch->getID();
	  MPI_Pack(&patchid, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  MPI_Pack(&m, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  ScatterRecord* record = pr.first->second;
	  int totalParticles=record->sendset->numParticles();
	  MPI_Pack(&totalParticles, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  total_reloc[1]+=totalParticles;
	  int datasize = datasizes[idx];
	  ASSERT(datasize>0);
	  MPI_Pack(&datasize, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());

	  int start = position;
	  ParticleSubset* pset = old_dw->getParticleSubset(matl, record->fromPatch);
	  ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
	  ParticleSubset* sendset=record->sendset;
	  posvar->packMPI(buf, sendsize, &position, pg, sendset, record->toPatch);
	  for(int v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);
	    var->packMPI(buf, sendsize, &position, pg, sendset);
	  }
	  int size=position-start;
	  if(size < datasize){
	    // MPI mis-esimated the size of the message.  For some
	    // reason, mpich does this all the time.  We must pad...
	    int diff=datasize-size;
	    char* junk = scinew char[diff];
	    MPI_Pack(junk, diff, MPI_CHAR, buf, sendsize, &position,
		     pg->getComm());
	    ASSERTEQ(position, start+datasize);
	    delete[] junk;
	  }
	  idx++;
	}
      }
    }
    ASSERT(position <= sendsize);
    ASSERT(sendsize > 0);    
    // Send (isend) the message
    MPI_Request rid;
    int to=iter->first;
    mpidbg << pg->myrank() << " Send relocate msg size " << sendsize << " tag " << RELOCATE_TAG << " to " << to << endl;
    MPI_Isend(buf, sendsize, MPI_PACKED, to, RELOCATE_TAG,
	      pg->getComm(), &rid);
    mpidbg << pg->myrank() << " done Sending relocate msg size " << sendsize << " tag " << RELOCATE_TAG << " to " << to << endl;
    sendbuffers.push_back(buf);
    sendrequests.push_back(rid);
  }

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
    MPI_Probe(iter->first, RELOCATE_TAG, pg->getComm(), &status);
    //ASSERT(status.MPI_ERROR == 0);      
    
    int size;
    MPI_Get_count(&status, MPI_PACKED, &size);
    ASSERT(size != 0);
    
    char* buf = scinew char[size];
    recvbuffers[idx]=buf;
    mpidbg << pg->myrank() << " Recv relocate msg size " << size << " tag " << RELOCATE_TAG << " from " << iter->first << endl;
    MPI_Recv(recvbuffers[idx], size, MPI_PACKED, iter->first,
	     RELOCATE_TAG, pg->getComm(), &status);

    mpidbg << pg->myrank() << " Done Recving relocate msg size " << size << " tag " << RELOCATE_TAG << " from " << iter->first << endl;
    // Partially unpack
    int position=0;
    int numrecords;
    MPI_Unpack(buf, size, &position, &numrecords, 1, MPI_INT,
	       pg->getComm());
    for(int i=0;i<numrecords;i++){
      int patchid;
      MPI_Unpack(buf, size, &position, &patchid, 1, MPI_INT,
		 pg->getComm());

      // find the patch from the id
      const Patch* toPatch = 0;
      for (int i = coarsestLevel->getIndex(); i < grid->numLevels(); i++) {
        LevelP checkLevel = grid->getLevel(i);
        int levelBaseID = checkLevel->getPatch(0)->getID();
        if (patchid >= levelBaseID && patchid < levelBaseID+checkLevel->numPatches()) {
          toPatch = checkLevel->getPatch(patchid-levelBaseID);
          break;
        }
      }
      ASSERT(toPatch != 0 && toPatch->getID() == patchid);
      int matl;
      MPI_Unpack(buf, size, &position, &matl, 1, MPI_INT,
		 pg->getComm());
      ASSERTRANGE(matl, 0, numMatls);
      int numParticles;
      MPI_Unpack(buf, size, &position, &numParticles, 1, MPI_INT,
		 pg->getComm());
      int datasize;
      MPI_Unpack(buf, size, &position, &datasize, 1, MPI_INT,
		 pg->getComm());
      char* databuf=buf+position;
      ASSERTEQ(lb->getPatchwiseProcessorAssignment(toPatch), me);
      scatter_records->saveRecv(toPatch, matl,
			       databuf, datasize, numParticles);
      position+=datasize;
      total_reloc[2]+=numParticles;
    }
  }
}

void Relocate::finalizeCommunication()
{
  // Wait to make sure that all of the sends completed
  int numsends = (int)sendrequests.size();
  vector<MPI_Status> statii(numsends);
  MPI_Waitall(numsends, &sendrequests[0], &statii[0]);

  // delete the buffers
  for(int i=0;i<(int)sendbuffers.size();i++)
    delete[] sendbuffers[i];
  for(int i=0;i<(int)recvbuffers.size();i++)
    if(recvbuffers[i])
      delete[] recvbuffers[i];
  sendrequests.clear();
  recvbuffers.clear();
  sendbuffers.clear();
}

const Patch* findFinePatch(const Point& pos, const Patch* guess, Level* fineLevel)
{
  if (guess && guess->getBox().contains(pos))
    return guess;
  return fineLevel->getPatchFromPoint(pos);
}

const Patch* findCoarsePatch(const Point& pos, const Patch* guess, Level* coarseLevel)
{
  if (guess && guess->getBox().contains(pos))
    return guess;
  return coarseLevel->getPatchFromPoint(pos);
}

void
Relocate::relocateParticles(const ProcessorGroup* pg,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw)
{
  if (patches->size() == 0) return;
  
  int me = pg->myrank();
  int total_reloc[3] = {0,0,0};

  // First pass: For each of the patches we own, look for particles
  // that left the patch.  Create a scatter record for each one.
  MPIScatterRecords scatter_records;
  int numMatls = (int)reloc_old_labels.size();
  Array2<ParticleSubset*> keepsets(patches->size(), numMatls);
  keepsets.initialize(0);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // AMR stuff
    const Level* curLevel = patch->getLevel();
    bool hasFiner   = curLevel->hasFinerLevel();
    bool hasCoarser = curLevel->hasCoarserLevel();
    Level* fineLevel=0;
    Level* coarseLevel=0;
    if(hasFiner){
      fineLevel = (Level*) curLevel->getFinerLevel().get_rep();
    }
    if(hasCoarser){
      coarseLevel = (Level*) curLevel->getCoarserLevel().get_rep();
    }

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Patch::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    // Find all of the neighbors, and add them to a set
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor=neighbors[i];
      scatter_records.addNeighbor(lb, pg, neighbor);
    }

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      constParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* keepset = scinew ParticleSubset(pset->getParticleSet(),
						      false, -1, 0,
						      pset->numParticles());
      ParticleSubset* delset = new_dw->getDeleteSubset(matl, patch);
      // Look for particles that left the patch, 
      // and if they are not in the delete set, put them in relocset

      ParticleSubset::iterator deliter = delset->begin();

      ASSERT(is_sorted(pset->begin(), pset->end()));
      ASSERT(is_sorted(delset->begin(), delset->end()));
      ASSERT(pset->begin() == pset->end() || *pset->begin() == 0);

      // when we find a relocated patch, check it against the next to-relocate particle
      //   before checking the rest of the patches
      const Patch* prevToRefinePatch=0, *prevToPatch=0, *prevToCoarsenPatch=0;
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	
	particleIndex idx = *iter;
        const Patch* toPatch = 0; // patch to relocate to

	if (deliter != delset->end() && idx == *deliter) {
  	  // all you need to do to keep a particle is neither keep it or 
          // relocate it.  So just go to the next deleted particle and wait for a match
	  deliter++;
	}
        else if (fineLevel && (toPatch = findFinePatch(px[idx], prevToRefinePatch, fineLevel)) != 0) {
          // do nothing - what we wanted was to set toPatch, and we'll add that to a scatterRecord
          prevToRefinePatch = toPatch;
        }
	else if(patch->getBox().contains(px[idx])){
          // is particle going to a finer patch?  Note, a particle does not have to leave the current patch
          // to go to a finer patch
	  keepset->addParticle(idx);
	}
	else {
          // not to delete or keep, so relocate it - add it to a scatter record
          if (prevToPatch && prevToPatch->getBox().contains(px[idx]))
            // optimization - check if particle went to the same patch as the previous relocated particle
            toPatch = prevToPatch;
          else {
            // This loop should change - linear searches are not good! However, since not very many particles leave the patches
	    // and there are a limited number of neighbors, perhaps it won't matter much
            int i=0;
            for(;i<(int)neighbors.size();i++){
	      if(neighbors[i]->getBox().contains(px[idx])){
	        break;
	      }
	    }
	    if(i == (int)neighbors.size()){
              //  Particle fell off of current level, maybe to a coarser one?
              if (coarseLevel) {
                toPatch = findCoarsePatch(px[idx], prevToCoarsenPatch, coarseLevel);
                prevToCoarsenPatch = toPatch;
              }
              if(!toPatch && level->containsPoint(px[idx])){
  	        // Make sure that the particle really left the world
                static ProgressiveWarning warn("A particle just travelled from one patch to another non-adjacent patch.  It has been deleted and we're moving on.",10);
                warn.invoke();
              }
            }
            else {
              toPatch = neighbors[i];
              prevToPatch = toPatch;
            }
          }
	}

        // now that we have toPatch - add it to a scatter record
        if (toPatch) {
          total_reloc[0]++;
	  ScatterRecord* record = scatter_records.findRecord(patch, toPatch, matl, pset);
	  record->sendset->addParticle(idx);
        }
      }

      if(keepset->numParticles() == pset->numParticles()){
	delete keepset;
	keepset=pset;
      }
      keepset->addReference();
      keepsets(p, m)=keepset;
    }
  }

  if (pg->size() > 1) {
    // send the particles where they need to go
    exchangeParticles(pg, patches, matls, old_dw, new_dw, &scatter_records, total_reloc);
  }

  // No go through each of our patches, and do the merge.  Also handle
  // the local case
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Patch::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      int numVars = (int)reloc_old_labels[m].size();
      vector<const Patch*> fromPatches;
      vector<ParticleSubset*> subsets;
      ParticleSubset* keepset = keepsets(p, m);
      ASSERT(keepset != 0);
      fromPatches.push_back(patch);
      subsets.push_back(keepset);
      for(int i=0;i<(int)neighbors.size();i++){
	const Patch* fromPatch=neighbors[i];
	int from = lb->getPatchwiseProcessorAssignment(fromPatch->getRealPatch());
	ASSERTRANGE(from, 0, pg->size());
	if(from == me){
	  ScatterRecord* record = scatter_records.findRecord(fromPatch, patch, matl);
	  if(record){
	    fromPatches.push_back(fromPatch);
	    subsets.push_back(record->sendset);
	  }
	}
      }
      MPIRecvBuffer* recvs = scatter_records.findRecv(patch, matl);
      map<const VarLabel*, ParticleVariableBase*>* newParts = 0;
      newParts = new_dw->getNewParticleState(matl,patch);
      bool adding_new_particles = false;
      if (newParts)
        adding_new_particles = true;
      ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, patch);
      if(recvs == 0 && subsets.size() == 1 && keepset == orig_pset && !adding_new_particles){
	// carry forward old data
	new_dw->saveParticleSubset(orig_pset, matl, patch);
	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	new_dw->put(*posvar, reloc_new_posLabel);
	for(int v=0;v<numVars;v++){
	  ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], orig_pset);
	  new_dw->put(*var, reloc_new_labels[m][v]);
	}
      } else {
        int numOldVariables = (int)subsets.size();
        if(newParts){
          map<const VarLabel*, ParticleVariableBase*>::iterator piter;
          piter = newParts->find(reloc_new_posLabel);
          if(piter == newParts->end())
            throw InternalError("didnt create new position", __FILE__, __LINE__);
          ParticleVariableBase* addedPos = piter->second;
          subsets.push_back(addedPos->getParticleSubset());
        }

	int totalParticles=0;
	for(int i=0;i<(int)subsets.size();i++)
	  totalParticles+=subsets[i]->numParticles();
	int numRemote=0;
	for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
	  numRemote+=buf->numParticles;
	}
	totalParticles+=numRemote;

	ParticleVariableBase* posvar = 
          new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	ParticleSubset* newsubset = 
          new_dw->createParticleSubset(totalParticles, matl, patch);

	// Merge local portion
	vector<ParticleVariableBase*> invars(subsets.size());
	for(int i=0;i<(int)numOldVariables;i++)
	  invars[i]=new_dw->getParticleVariable(reloc_old_posLabel, matl,
						fromPatches[i]);
        if(newParts){
          map<const VarLabel*, ParticleVariableBase*>::iterator piter;
          piter = newParts->find(reloc_new_posLabel);
          if(piter == newParts->end())
            throw InternalError("didnt create new position", __FILE__, __LINE__);
          ParticleVariableBase* addedPos = piter->second;
          invars[subsets.size()-1] = addedPos;
          fromPatches.push_back(patch);
        }
	ParticleVariableBase* newpos = posvar->clone();
	newpos->gather(newsubset, subsets, invars, fromPatches, numRemote);

	vector<ParticleVariableBase*> vars(numVars);
	for(int v=0;v<numVars;v++){
	  const VarLabel* label = reloc_old_labels[m][v];
	  ParticleVariableBase* var = new_dw->getParticleVariable(label, orig_pset);
	  for(int i=0;i<numOldVariables;i++)
	    invars[i]=new_dw->getParticleVariable(label, matl, fromPatches[i]);
          if(newParts){
            map<const VarLabel*, ParticleVariableBase*>::iterator piter;
            piter = newParts->find(reloc_new_labels[m][v]);
            if(piter == newParts->end()) {
              cout << "reloc_new_labels = " << reloc_new_labels[m][v]->getName()
                   << endl;
              throw InternalError("didnt create new variable of this type", __FILE__, __LINE__);
            }
            ParticleVariableBase* addedVar = piter->second;
            invars[subsets.size()-1] = addedVar;
          }
	  ParticleVariableBase* newvar = var->clone();
	  newvar->gather(newsubset, subsets, invars, fromPatches, numRemote);
	  vars[v]=newvar;
	}
	// Unpack MPI portion
	particleIndex idx = totalParticles-numRemote;
	for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
	  int position=0;
	  ParticleSubset* unpackset = scinew ParticleSubset(newsubset->getParticleSet(),
							    false, matl, patch, 0);
	  unpackset->resize(buf->numParticles);
	  for(int p=0;p<buf->numParticles;p++,idx++)
	    unpackset->set(p, idx);
	  newpos->unpackMPI(buf->databuf, buf->bufsize, &position,
			    pg, unpackset);
	  for(int v=0;v<numVars;v++)
	    vars[v]->unpackMPI(buf->databuf, buf->bufsize, &position,
			       pg, unpackset);
	  ASSERT(position <= buf->bufsize);
	  delete unpackset;
	}
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
      }
      if(keepset->removeReference())
	delete keepset;
    }
  }

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "total_reloc: " << total_reloc[0] << ", " << total_reloc[1]
	       << ", " << total_reloc[2] << "\n";
    cerrLock.unlock();
  }

  if(!mixedDebug.active()){
    // this is bad for the MixedScheduler... I think it is ok to
    // just remove it... at least for now... as it is only for info
    // and debug purposes...
    // Communicate the number of particles to processor zero, and
    // print them out
    int alltotal[3] = {total_reloc[0], total_reloc[1], total_reloc[2] };

    // don't reduce if number of patches on this level is < num procs.  Will wait forever in reduce.
    //if (!lb->isDynamic() && level->getGrid()->numLevels() == 1 && level->numPatches() >= pg->size() && pg->size() > 1) {
    if (pg->size() > 1) {
      mpidbg << pg->myrank() << " Relocate reduce\n";
      MPI_Reduce(total_reloc, &alltotal, 3, MPI_INT, MPI_SUM, 0,
                pg->getComm());
      mpidbg << pg->myrank() << " Done Relocate reduce\n";
    }
    if(pg->myrank() == 0){
      ASSERTEQ(alltotal[1], alltotal[2]);
      if(alltotal[0] != 0)
        cerr << "Particles crossing patch boundaries: " << alltotal[0] << ", crossing processor boundaries: " << alltotal[1] << '\n';
    }
  }

  if (pg->size() > 1)
    finalizeCommunication();

} // end relocateParticles()

