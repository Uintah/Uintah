

#include <Packages/Uintah/CCA/Components/Schedulers/ParticleLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Thread/Time.h>

#include <iostream> // debug only

using namespace Uintah;

using std::cerr;

#define DAV_DEBUG 0

class PairCompare {
public:
  inline bool operator()(const pair<particleIndex,int> p1, 
			 const pair<particleIndex,int> p2) const {
    return p1.first < p2.first;
  }
private:
};


ParticleLoadBalancer::ParticleLoadBalancer(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
  //need to compensate for restarts
  d_currentTime = 0;
  d_currentTimestep = 0;
  
  d_state = needLoadBalance;
  d_do_AMR = false;
  d_pspec = 0;
}

ParticleLoadBalancer::~ParticleLoadBalancer()
{
}

void ParticleLoadBalancer::assignPatches2(const LevelP& level, 
					 const ProcessorGroup* group,
					 const Scheduler* sch)
{

  // if this is just during a recompile to compensate for 
  // loadbalancing on the last timestep
  if (d_state != needLoadBalance) {
    d_oldAssignment = d_processorAssignment;
    d_state = idle;
    return;
  }

  // this assigns patches in a cyclic form - every time we re-load balance
  // we move each patch up one proc - this obviously isn't a very good
  // lb technique, but it tests its capabilities pretty well.

  SchedulerCommon* sc = (SchedulerCommon*)(sch);
  int numProcs = group->size();
  int numPatches = level->numPatches();
  int myrank = group->myrank();

  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>
    (sc->get_dw(0));


  if (dw == 0) { // first dw
    d_processorAssignment.resize(numPatches);

    for (Level::const_patchIterator iter = level->patchesBegin(); 
	 iter != level->patchesEnd(); iter++) 
    {
      Patch *patch = *iter;
      d_processorAssignment[patch->getID()] = patch->getID() % numProcs;
    }

    d_oldAssignment = d_processorAssignment;
  }
  else {
    d_oldAssignment = d_processorAssignment;
    for (int i = 0; i < numPatches; i++) {
      d_processorAssignment[i] = (d_processorAssignment[i] + 1 ) % numProcs;
    }
  }
  for (int i = 0; i < numPatches; i++) {
    if (myrank == 0)
      cout << "Patch " << i << " -> proc " << d_processorAssignment[i] << endl;
  }

  d_state = postLoadBalance;
}

void ParticleLoadBalancer::assignPatches(const LevelP& level, 
					 const ProcessorGroup* group,
					 const Scheduler* sch)
{
  // if this is just during a recompile to compensate for 
  // loadbalancing on the last timestep
  if (d_state != needLoadBalance) {
    d_oldAssignment = d_processorAssignment;
    d_state = idle;
    return;
  }

  double time = Time::currentSeconds();

  unsigned int numProcs = group->size();
  unsigned int numPatches = level->numPatches();
  unsigned int myrank = group->myrank();
  unsigned int i;

  //setup for get each patch's number of particles
  numPatches = level->numPatches();
  vector<pair<particleIndex, int> > particleList;

  // get how many particles were each patch had at the end of the last timestep
  SchedulerCommon* sc = (SchedulerCommon*)(sch);

  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>
    (sc->get_dw(0));

  if (dw == 0) { // first dw
    d_processorAssignment.resize(numPatches);
    d_oldAssignment.resize(numPatches);
  }
  else {
    d_oldAssignment = d_processorAssignment; // copy the current to the old
  }

  // proc 0 - order patches by processor #
  vector<int> sorted_processorAssignment = d_processorAssignment;
  sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());

  vector<int> displs;
  vector<int> recvcounts(numProcs,0); // init the counts to 0

  int offsetProc = 0;
  for (i = 0; i < d_processorAssignment.size(); i++) {
    // set the offsets for the MPI_Gatherv
    if ( offsetProc == sorted_processorAssignment[i]) {
      displs.push_back(i);
      offsetProc++;
    }
    recvcounts[sorted_processorAssignment[i]]++;
  }

  // find out how many particles per patch, and store that number
  // along with the patch number in particleList
  for (Level::const_patchIterator iter = level->patchesBegin(); 
       iter != level->patchesEnd(); iter++) 
  {
    Patch *patch = *iter;
    int id = patch->getID();
    if (d_processorAssignment[id] != myrank)
      continue;

    int thisPatchParticles = 0;

    if (dw)
    {
      //loop through the materials and add up the particles
      const MaterialSet *matls = sc->getMaterialSet();
      const MaterialSubset *ms;
      if (matls) {
	ms = sc->getMaterialSet()->getSubset(0);
	int size = ms->size();
	for (int matl = 0; matl < size; matl++) {
	  ParticleSubset* psubset = dw->getParticleSubset(matl, patch);
	  if (psubset)
	    thisPatchParticles += psubset->numParticles();
	}
      }
    }
    // add to particle list
    pair<particleIndex,int> p(thisPatchParticles,id);
    particleList.push_back(p);

  }

  // each proc - gather all particles together
  vector<pair<particleIndex,int> > allParticles(numPatches);

  //construct a mpi datatype for a pair
  MPI_Datatype pairtype;
  MPI_Type_contiguous(2, MPI_INT, &pairtype);
  MPI_Type_commit(&pairtype);

  MPI_Gatherv(&particleList[0],particleList.size(),pairtype,
	      &allParticles[0], &recvcounts[0], &displs[0], pairtype,
	      0, group->getComm());

  
  if (myrank == 0) {
    //cout << "Post gather\n";
    //for (i = 0; i < allParticles.size(); i++)
    //cout << "Patch: " << allParticles[i].second
    //<< " -> " << allParticles[i].first << endl;
  }
  // proc 0 - associate patches to particles, load balance, 
  //   MPI_Bcast it to all procs
  


  d_processorAssignment.clear();
  d_processorAssignment.resize(numPatches);




  if (myrank == 0) {
    // sort the particle list.  We need the patch number in there as 
    // well in order to look up the patch number later
    sort(allParticles.begin(), allParticles.end(), PairCompare());
    
    int patchesLeft = numPatches;
    int minPatch = 0;
    int maxPatch = numPatches-1;
    
    // assignment algorithm: loop through the processors (repeatedly) until we
    //   have no more patches to assign.  If, there are twice as many (or more)
    //   patches than processors, than assign the patch with the most and the
    //   the least particles to the current processor.  Otherwise, assign the 
    //   the patch with the most particles to the current processor.
    while (patchesLeft > 0) {
      for (i = 0; i < numProcs; i++) {
	int highpatch = allParticles[maxPatch].second;
	int lowpatch = allParticles[minPatch].second;
	if (patchesLeft >= 2*(numProcs-i)) {
	  // give it min and max
	  d_processorAssignment[highpatch]=i;
	  maxPatch--;
	  d_processorAssignment[lowpatch]=i;
	  minPatch++;
	  patchesLeft -= 2;
	  
	} else if (patchesLeft > 0) {
	  //give it max
	  d_processorAssignment[maxPatch]=i;
	  maxPatch--;
	  patchesLeft--;
	} else if (patchesLeft == 0) {
	  break;
	}
      }
    }

  }
  MPI_Bcast(&d_processorAssignment[0], numPatches, MPI_INT,0,group->getComm());

//   if (!myrank)
//     for (i = 0; i < numPatches; i++) {
//       cout << "On patch " << allParticles[i].second << " There are " 
// 	   << allParticles[i].first << "particles" << endl;
//     }
  for (i = 0; i < numPatches; i++) {
    cout << "Patch " << i << "->proc " << d_processorAssignment[i] << endl;
  }
  if (!myrank)
    cout << "particle sorting took " << Time::currentSeconds() - time << "s\n";

  // if first DW, assign old to current processor assignment
  if (!dw) {
    d_oldAssignment = d_processorAssignment;
  }

  d_state = postLoadBalance;

}

void ParticleLoadBalancer::assignResources(DetailedTasks& graph,
					     const ProcessorGroup* group)
{
  int nTasks = graph.numTasks();
  int numProcs = group->size();

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);
    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);
      int idx = getPatchwiseProcessorAssignment(patch,group);
      task->assignResource(idx);
      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = getPatchwiseProcessorAssignment(p,group);
	if(pidx != idx){
	  cerr << "WARNING: inconsistent task assignment in ParticleLoadBalancer\n";
	}
      }
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
	task->assignResource( Parallel::getRootProcessorGroup()->myrank() );
      } else if( task->getTask()->getType() == Task::InitialSend){
	// Already assigned, do nothing
	ASSERT(task->getAssignedResourceIndex() != -1);
      } else {
#if DAV_DEBUG
	cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
	task->assignResource(0);
      }
    }
  }
}

int
ParticleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
						      const ProcessorGroup* group)
{
  int proc = d_processorAssignment[patch->getID()];
  //cout << group->myrank() << " Requesting patch " << patch->getID()
  //   << " which is stored on processor " << proc << endl;
  //int proc = (patch->getLevelIndex()*numProcs)/patch->getLevel()->numPatches();
  ASSERTRANGE(proc, 0, group->size());
  return proc;
}

int
ParticleLoadBalancer::getOldProcessorAssignment(const VarLabel* var, 
						const Patch* patch, 
						const int matl, 
						const ProcessorGroup* group)
{
  if (var->typeDescription()->isReductionVariable()) {
    return group->myrank();
  }
  int proc = d_oldAssignment[patch->getID()];
  //cout << group->myrank() << " Requesting patch " << patch->getID()
  //   << " which *used to be* stored on processor " << proc << endl;
  ASSERTRANGE(proc, 0, group->size());
  return proc;
  //return getPatchwiseProcessorAssignment(patch, group);
}

bool 
ParticleLoadBalancer::needRecompile(double time, double delt, 
				    const GridP& grid)
{
  //need to compensate for restarts
  d_currentTime += delt;
  d_currentTimestep++;

  if (d_lbTimestepInterval != 0 && 
      d_currentTimestep >= d_lastLbTimestep + d_lbTimestepInterval) {
    d_lastLbTimestep = d_currentTimestep;
    d_state = needLoadBalance;
  }
  else if (d_lbInterval != 0 && d_currentTime >= d_lastLbTime + d_lbInterval) {
    d_lastLbTime = d_currentTime;
    d_state = needLoadBalance;
  }
  
  return d_state; // to recompile when need to load balance or post lb
} 
void 
ParticleLoadBalancer::problemSetup(ProblemSpecP& pspec)
{
   ProblemSpecP p = pspec->findBlock("LoadBalancer");
   if (p != 0) {
     if(!p->get("timestepInterval", d_lbTimestepInterval))
       d_lbTimestepInterval = 0;
     if(!p->get("interval", d_lbInterval)
	&& d_lbTimestepInterval == 0)
       d_lbInterval = 0.0; // default
   }
   else {
     d_lbTimestepInterval = 0;
     d_lbInterval = 0.0;
   }

   d_pspec = pspec;
}

const PatchSet*
ParticleLoadBalancer::createPerProcessorPatchSet(const LevelP& level,
						   const ProcessorGroup* world)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(world->size());
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch, world);
    ASSERTRANGE(proc, 0, world->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  patches->sortSubsets();
  return patches;
}


void
ParticleLoadBalancer::createNeighborhood(const GridP& grid,
					 const ProcessorGroup* group,
					 const Scheduler* sch)
{
  int me = group->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  d_neighbors.clear();
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    assignPatches2(level, group, sch);
    //d_processorAssignment.push_back(0);
    //d_processorAssignment.push_back(1);
    for(Level::const_patchIterator iter = level->patchesBegin();
	iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      if(getPatchwiseProcessorAssignment(patch, group) == me){
	Level::selectType n;
	IntVector lowIndex, highIndex;
	patch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0),
				      Ghost::AroundCells, maxGhost, n,
				      lowIndex, highIndex);
	for(int i=0;i<(int)n.size();i++){
	  const Patch* neighbor = n[i];
	  if(d_neighbors.find(neighbor) == d_neighbors.end())
	    d_neighbors.insert(neighbor);
	}
      }
    }
  }
}

bool
ParticleLoadBalancer::inNeighborhood(const PatchSubset* ps,
				       const MaterialSubset*)
{
  for(int i=0;i<ps->size();i++){
    const Patch* patch = ps->get(i);
    if(d_neighbors.find(patch) != d_neighbors.end())
      return true;
  }
  return false;
}

bool
ParticleLoadBalancer::inNeighborhood(const Patch* patch)
{
  if(d_neighbors.find(patch) != d_neighbors.end())
    return true;
  else
    return false;
}
