
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>

#include <Core/Containers/Array2.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

#include <sci_algorithm.h>

#include <map>
#include <set>

#define RELOCATE_TAG            0x3fff

using namespace std;
using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

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

MPIRelocate::MPIRelocate()
{
}

MPIRelocate::~MPIRelocate()
{
}

SPRelocate::SPRelocate()
{
}

SPRelocate::~SPRelocate()
{
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

  class SPScatterRecords {
  public:
    typedef set<ScatterRecord*, CompareScatterRecord> settype;
    settype records;

    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl,
			      ParticleSubset* pset);
    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl);

    ~SPScatterRecords();
  };
  struct MPIScatterProcessorRecord {
    typedef vector<const Patch*> patchestype;
    patchestype patches;
    void sortPatches();
  };
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
  class MPIScatterRecords {
  public:
    // map the to patch and matl to the ScatterRecord
    typedef multimap<pair<const Patch*, int>, ScatterRecord*> maptype;
    maptype records;
    
    typedef map<int, MPIScatterProcessorRecord*> procmaptype;
    procmaptype procs;

    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl,
			      ParticleSubset* pset);
    ScatterRecord* findRecord(const Patch* from, const Patch* to, int matl);
    void addNeighbor(LoadBalancer* lb, const ProcessorGroup* pg,
		     const Patch* to);

    typedef map<pair<const Patch*, int>, MPIRecvBuffer*> recvmaptype;
    recvmaptype recvs;
    void saveRecv(const Patch* to, int matl,
		  char* databuf, int bufsize, int numParticles);
    MPIRecvBuffer* findRecv(const Patch* to, int matl);

    ~MPIScatterRecords();
  };
} // End namespace Uintah

ScatterRecord* SPScatterRecords::findRecord(const Patch* from,
					    const Patch* to, int matl,
					    ParticleSubset* pset)
{
  ScatterRecord* rec = scinew ScatterRecord(from, to, matl);
  settype::iterator iter = records.find(rec);
  if(iter == records.end()){
    rec->sendset = scinew ParticleSubset(pset->getParticleSet(), false, -1, 0, 0);
    records.insert(rec);
    return rec;
  } else {
    delete rec;
    return *iter;
  }
}

ScatterRecord* SPScatterRecords::findRecord(const Patch* from,
					    const Patch* to, int matl)
{
  ScatterRecord rec(from, to, matl);  
  settype::iterator iter = records.find(&rec);
  if(iter == records.end())
    return 0;
  else
    return *iter;
}

SPScatterRecords::~SPScatterRecords()
{
  for(settype::iterator iter = records.begin(); iter != records.end(); iter++){
    delete (*iter)->sendset;
    delete *iter;
  }
}

void
SPRelocate::relocateParticles(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  int total_reloc=0;
  int check_total_reloc=0;

  // First pass: For each of the patches we own, look for particles
  // that left the patch.  Create a scatter record for each one.
  SPScatterRecords scatter_records;
  int numMatls = (int)reloc_old_labels.size();
  Array2<ParticleSubset*> keepsets(patches->size(), numMatls);
  keepsets.initialize(0);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = new_dw->getDeleteSubset(matl, patch);
      constParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* relocset = scinew ParticleSubset(pset->getParticleSet(),
						       false, -1, 0, 0);
      ParticleSubset* keepset = scinew ParticleSubset(pset->getParticleSet(),
						      false, -1, 0,
						      pset->numParticles());

      // Look for particles that left the patch, 
      // and if they are not in the delete set, put them in relocset

      ParticleSubset::iterator deliter = delset->begin();
      
      ASSERT(is_sorted(pset->begin(), pset->end()));
      ASSERT(is_sorted(delset->begin(), delset->end()));
      ASSERT(pset->begin() == pset->end() || *pset->begin() == 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){

	bool keep = true; 
	particleIndex idx = *iter;

	// if particle is in delete set, don't keep or relocate
	//	for (ParticleSubset::iterator deliter = delset->begin(); 
	//	     deliter != delset->end(); deliter++) {

	if (deliter != delset->end() && idx == *deliter) {
	  keep = false;
	  deliter++;
	}

	if(patch->getBox().contains(px[idx]) && keep){
	  keepset->addParticle(idx);
	}
	else if(keep) {
	  relocset->addParticle(idx);
	}
      }

      if(keepset->numParticles() == pset->numParticles()){
	delete keepset;
	keepset=pset;
      }
      keepset->addReference();
      keepsets(p, m)=keepset;

      if(relocset->numParticles() > 0){
	total_reloc+=relocset->numParticles();
	// Figure out exactly where they went...
	for(ParticleSubset::iterator iter = relocset->begin();
	    iter != relocset->end(); iter++){
	  particleIndex idx = *iter;
	  // This loop should change - linear searches are not good!
	  // However, since not very many particles leave the patches
	  // and there are a limited number of neighbors, perhaps it
	  // won't matter much
	  int i;
	  for(i=0;i<(int)neighbors.size();i++){
	    if(neighbors[i]->getBox().contains(px[idx])){
	      break;
	    }
	  }
	  if(i == (int)neighbors.size()){
	    // Make sure that the particle really left the world
	    if(level->containsPoint(px[idx]))
	      SCI_THROW(InternalError("Particle fell through the cracks!"));
	  } else {
	    // Save this particle set for sending later
	    const Patch* toPatch=neighbors[i];
	    ScatterRecord* record = scatter_records.findRecord(patch,
							       toPatch,
							       matl, pset);
	    record->sendset->addParticle(idx);

	    // Optimization: see if other (consecutive) particles
	    // also went to this same patch
	    ParticleSubset::iterator iter2=iter;
	    iter2++;
	    for(;iter2 != relocset->end(); iter2++){
	      particleIndex idx2 = *iter2;
	      if(toPatch->getBox().contains(px[idx2])){
		iter++;
		record->sendset->addParticle(idx2);
	      } else {
		break;
	      }
	    }
	  }
	}
      }
      delete relocset;
    }
  }

  // No go through each of our patches, and do the merge.  Also handle
  // the local case
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ASSERT(!patch->isVirtual());
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      int numVars = (int)reloc_old_labels[matl].size();
      vector<const Patch*> fromPatches;
      vector<ParticleSubset*> subsets;
      ParticleSubset* keepset = keepsets(p, m);
      ASSERT(keepset != 0);
      fromPatches.push_back(patch);
      subsets.push_back(keepset);
      for(int i=0;i<(int)neighbors.size();i++){
	const Patch* fromPatch=neighbors[i];
	ScatterRecord* record = scatter_records.findRecord(fromPatch,
							   patch, matl);
	if(record){
	  fromPatches.push_back(fromPatch);
	  check_total_reloc += record->sendset->numParticles();
	  subsets.push_back(record->sendset);
	}
      }
      ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, patch);
      if(subsets.size() == 1 && keepset == orig_pset){
	// carry forward old data
	new_dw->saveParticleSubset(matl, patch, orig_pset);
	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	new_dw->put(*posvar, reloc_new_posLabel);
	for(int v=0;v<numVars;v++){
	  ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], orig_pset);
	  new_dw->put(*var, reloc_new_labels[matl][v]);
	}
      } else {
	int totalParticles=0;
	for(int i=0;i<(int)subsets.size();i++)
	  totalParticles+=subsets[i]->numParticles();

	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, matl, patch);

	// Merge local portion
	vector<ParticleVariableBase*> invars(subsets.size());
	for(int i=0;i<(int)subsets.size();i++)
	  invars[i]=new_dw->getParticleVariable(reloc_old_posLabel, matl,
						fromPatches[i]);
	ParticleVariableBase* newpos = posvar->clone();
	newpos->gather(newsubset, subsets, invars, fromPatches, 0);

	vector<ParticleVariableBase*> vars(numVars);
	for(int v=0;v<numVars;v++){
	  const VarLabel* label = reloc_old_labels[matl][v];
	  ParticleVariableBase* var = new_dw->getParticleVariable(label, orig_pset);
	  for(int i=0;i<(int)subsets.size();i++)
	    invars[i]=new_dw->getParticleVariable(label, matl, fromPatches[i]);
	  ParticleVariableBase* newvar = var->clone();
	  newvar->gather(newsubset, subsets, invars, fromPatches, 0);
	  vars[v]=newvar;
	}
	// Put the data back in the data warehouse
	new_dw->put(*newpos, reloc_new_posLabel);
	delete newpos;
	for(int v=0;v<numVars;v++){
	  new_dw->put(*vars[v], reloc_new_labels[matl][v]);
	  delete vars[v];
	}
      }
      if(keepset->removeReference())
	delete keepset;
    }
  }

  ASSERTEQ(total_reloc, check_total_reloc);
  if(total_reloc != 0)
    cerr << "Particles crossing patch boundaries: " << total_reloc << '\n';
}

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
  ASSERTEQ(numMatls, matls->getSubset(0)->size());
  for (int m = 0; m< numMatls; m++)
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());
  Task* t = scinew Task("Relocate::relocateParticles",
			this, &Relocate::relocateParticles);
  if(lb)
    t->usesMPI();
  t->requires( Task::NewDW, old_posLabel, Ghost::None);
  for(int m=0;m < numMatls;m++){
    MaterialSubset* thismatl = scinew MaterialSubset();
    thismatl->add(m);
    for(int i=0;i<(int)old_labels[m].size();i++)
      t->requires( Task::NewDW, old_labels[m][i], thismatl, Ghost::None);

    t->computes( new_posLabel, thismatl);
    for(int i=0;i<(int)new_labels[m].size();i++)
      t->computes(new_labels[m][i], thismatl);
  }
  const PatchSet* patches;
  if(lb)
    patches = lb->createPerProcessorPatchSet(level, pg);
  else
    patches = level->allPatches();
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
  int toProc = lb->getPatchwiseProcessorAssignment(neighbor, pg);
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
MPIRelocate::relocateParticles(const ProcessorGroup* pg,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw)
{
  int total_reloc[3], v;
  total_reloc[0]=total_reloc[1]=total_reloc[2]=0;
  typedef MPIScatterRecords::maptype maptype;
  typedef MPIScatterRecords::procmaptype procmaptype;
  typedef MPIScatterProcessorRecord::patchestype patchestype;

  // First pass: For each of the patches we own, look for particles
  // that left the patch.  Create a scatter record for each one.
  MPIScatterRecords scatter_records;
  int numMatls = (int)reloc_old_labels.size();
  Array2<ParticleSubset*> keepsets(patches->size(), numMatls);
  keepsets.initialize(0);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
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

      ParticleSubset* relocset = scinew ParticleSubset(pset->getParticleSet(),
						       false, -1, 0, 0);
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

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	
	bool keep = true; 
	particleIndex idx = *iter;

	// if particle is in delete set, don't keep or relocate
	//	for (ParticleSubset::iterator deliter = delset->begin(); 
	//	     deliter != delset->end(); deliter++) {

	if (deliter != delset->end() && idx == *deliter) {
	  keep = false;
	  deliter++;
	}
	if(patch->getBox().contains(px[idx]) && keep){
	  keepset->addParticle(idx);
	}
	else if(keep) {
	  relocset->addParticle(idx);
	}
      }

      if(keepset->numParticles() == pset->numParticles()){
	delete keepset;
	keepset=pset;
      }
      keepset->addReference();
      keepsets(p, m)=keepset;

      if(relocset->numParticles() > 0){
	total_reloc[0]+=relocset->numParticles();
	// Figure out exactly where they went...
	for(ParticleSubset::iterator iter = relocset->begin();
	    iter != relocset->end(); iter++){
	  particleIndex idx = *iter;
	  // This loop should change - linear searches are not good!
	  // However, since not very many particles leave the patches
	  // and there are a limited number of neighbors, perhaps it
	  // won't matter much
	  int i;
	  for(i=0;i<(int)neighbors.size();i++){
	    if(neighbors[i]->getBox().contains(px[idx])){
	      break;
	    }
	  }
	  if(i == (int)neighbors.size()){
	    // Make sure that the particle really left the world
	    if(level->containsPoint(px[idx]))
	      SCI_THROW(InternalError("Particle fell through the cracks!"));
	  } else {
	    // Save this particle set for sending later
	    const Patch* toPatch=neighbors[i];
	    ScatterRecord* record = scatter_records.findRecord(patch,
							       toPatch,
							       matl, pset);
	    record->sendset->addParticle(idx);

	    // Optimization: see if other (consecutive) particles
	    // also went to this same patch
	    ParticleSubset::iterator iter2=iter;
	    iter2++;
	    for(;iter2 != relocset->end(); iter2++){
	      particleIndex idx2 = *iter2;
	      if(toPatch->getBox().contains(px[idx2])){
		iter++;
		record->sendset->addParticle(idx2);
	      } else {
		break;
	      }
	    }
	  }
	}
      }
      delete relocset;
    }
  }

  int me = pg->myrank();
  vector<char*> sendbuffers;
  vector<MPI_Request> sendrequests;
  for(procmaptype::iterator iter = scatter_records.procs.begin();
      iter != scatter_records.procs.end(); iter++){
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
      for(int matl=0;matl<numMatls;matl++){
	int numVars = (int)reloc_old_labels[matl].size();
	int numParticles=0;
	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records.records.equal_range(make_pair(toPatch, matl));
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
	  for(v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], pset);
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
      for(int matl=0;matl<numMatls;matl++){
	int numVars = (int)reloc_old_labels[matl].size();

	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records.records.equal_range(make_pair(toPatch, matl));
	for(;pr.first != pr.second; pr.first++){
	  int patchid = toPatch->getID();
	  MPI_Pack(&patchid, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  MPI_Pack(&matl, 1, MPI_INT, buf, sendsize, &position,
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
	  for(v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], pset);
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
    MPI_Isend(buf, sendsize, MPI_PACKED, to, RELOCATE_TAG,
	      pg->getComm(), &rid);
    sendbuffers.push_back(buf);
    sendrequests.push_back(rid);
  }

  // Receive, and handle the local case too...
  // Foreach processor, post a receive
  vector<char*> recvbuffers(scatter_records.procs.size());

  // I wish that there was an Iprobe_some call, so that we could do
  // this more dynamically...
  int idx=0;
  for(procmaptype::iterator iter = scatter_records.procs.begin();
      iter != scatter_records.procs.end(); iter++, idx++){
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
    MPI_Recv(recvbuffers[idx], size, MPI_PACKED, iter->first,
	     RELOCATE_TAG, pg->getComm(), &status);

    // Partially unpack
    int position=0;
    int numrecords;
    MPI_Unpack(buf, size, &position, &numrecords, 1, MPI_INT,
	       pg->getComm());
    for(int i=0;i<numrecords;i++){
      int patchid;
      MPI_Unpack(buf, size, &position, &patchid, 1, MPI_INT,
		 pg->getComm());
      const Patch* toPatch = Patch::getByID(patchid);
      ASSERT(toPatch != 0);
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
      ASSERTEQ(lb->getPatchwiseProcessorAssignment(toPatch, pg), me);
      scatter_records.saveRecv(toPatch, matl,
			       databuf, datasize, numParticles);
      position+=datasize;
      total_reloc[2]+=numParticles;
    }
  }

  // No go through each of our patches, and do the merge.  Also handle
  // the local case
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      int numVars = (int)reloc_old_labels[matl].size();
      vector<const Patch*> fromPatches;
      vector<ParticleSubset*> subsets;
      ParticleSubset* keepset = keepsets(p, m);
      ASSERT(keepset != 0);
      fromPatches.push_back(patch);
      subsets.push_back(keepset);
      for(int i=0;i<(int)neighbors.size();i++){
	const Patch* fromPatch=neighbors[i];
	int from = lb->getPatchwiseProcessorAssignment(fromPatch->getRealPatch(), pg);
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
      ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, patch);
      if(recvs == 0 && subsets.size() == 1 && keepset == orig_pset){
	// carry forward old data
	new_dw->saveParticleSubset(matl, patch, orig_pset);
	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	new_dw->put(*posvar, reloc_new_posLabel);
	for(v=0;v<numVars;v++){
	  ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], orig_pset);
	  new_dw->put(*var, reloc_new_labels[matl][v]);
	}
      } else {
	int totalParticles=0;
	for(int i=0;i<(int)subsets.size();i++)
	  totalParticles+=subsets[i]->numParticles();
	int numRemote=0;
	for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
	  numRemote+=buf->numParticles;
	}
	totalParticles+=numRemote;

	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, matl, patch);

	// Merge local portion
	vector<ParticleVariableBase*> invars(subsets.size());
	for(int i=0;i<(int)subsets.size();i++)
	  invars[i]=new_dw->getParticleVariable(reloc_old_posLabel, matl,
						fromPatches[i]);
	ParticleVariableBase* newpos = posvar->clone();
	newpos->gather(newsubset, subsets, invars, fromPatches, numRemote);

	vector<ParticleVariableBase*> vars(numVars);
	for(v=0;v<numVars;v++){
	  const VarLabel* label = reloc_old_labels[matl][v];
	  ParticleVariableBase* var = new_dw->getParticleVariable(label, orig_pset);
	  for(int i=0;i<(int)subsets.size();i++)
	    invars[i]=new_dw->getParticleVariable(label, matl, fromPatches[i]);
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
	  for(v=0;v<numVars;v++)
	    vars[v]->unpackMPI(buf->databuf, buf->bufsize, &position,
			       pg, unpackset);
	  ASSERT(position <= buf->bufsize);
	  delete unpackset;
	}
	ASSERTEQ(idx, totalParticles);

#if 0
	for(v=0;v<numVars;v++){
	  const VarLabel* label = reloc_new_labels[matl][v];
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
	for(v=0;v<numVars;v++){
	  new_dw->put(*vars[v], reloc_new_labels[matl][v]);
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
    int alltotal[3];

    MPI_Reduce(total_reloc, &alltotal, 3, MPI_INT, MPI_SUM, 0,
	       pg->getComm());
    if(pg->myrank() == 0){
      ASSERTEQ(alltotal[1], alltotal[2]);
      if(alltotal[0] != 0)
	cerr << "Particles crossing patch boundaries: " << alltotal[0] << ", crossing processor boundaries: " << alltotal[1] << '\n';
    }
  }

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

} // end relocateParticles()

