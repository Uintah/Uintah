
#include <Packages/Uintah/CCA/Components/Schedulers/LoadBalancerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/ParticleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <sstream>

using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

LoadBalancerCommon::LoadBalancerCommon(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

LoadBalancerCommon::~LoadBalancerCommon()
{
}

void LoadBalancerCommon::assignResources(DetailedTasks& graph)
{
  int nTasks = graph.numTasks();

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Assigning Tasks to Resources!\n";
    cerrLock.unlock();
  }

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);

    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->size());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource((idx/d_outputNthProc)*d_outputNthProc);
      }
      else        
        task->assignResource(idx);

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "1) Task " << *(task->getTask()) << " put on resource "
		   << idx << "\n";
	cerrLock.unlock();
      }

      ostringstream ostr;
      ostr << patch->getID() << ':' << idx;

      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = getPatchwiseProcessorAssignment(p);
        ostr << ' ' << p->getID() << ';' << pidx;
	ASSERTRANGE(pidx, 0, d_myworld->size());
	if(pidx != idx){
	  cerrLock.lock();
	  cerr << d_myworld->myrank() << " WARNING: inconsistent task (" << task->getTask()->getName() 
	       << ") assignment (" << pidx << ", " << idx 
	       << ") in LoadBalancerCommon\n";
	  cerrLock.unlock();
	}
      }
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
	task->assignResource( d_myworld->myrank() );

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "  Resource (for no patch task) is : " 
		     << d_myworld->myrank() << "\n";
	  cerrLock.unlock();
	}

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
    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << "For Task: " << *task << "\n";
      cerrLock.unlock();
    }

  }

  //ParticleLoadBalancer* plb = (ParticleLoadBalancer*) this;
  //for (int i = 0; i < d_myworld->size(); i++) {
  //cout << d_myworld->myrank() << " Patch " << i
  //     << " -> proc " << plb->d_processorAssignment[i] << " old: "
  //     << plb->d_oldAssignment[i] << endl;
  //}
}

const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet(const LevelP& level)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->size());
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch);
    ASSERTRANGE(proc, 0, d_myworld->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  patches->sortSubsets();  
  return patches;
}

void
LoadBalancerCommon::createNeighborhood(const GridP& grid)
{
  int me = d_myworld->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  d_neighbors.clear();

  // go through all patches on all levels, and if the patchwise
  // processor assignment equals the current processor, then store the 
  // patch's neighbors in the load balancer array
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);

    for(Level::const_patchIterator iter = level->patchesBegin();
	iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;

      // we need to check both where the patch is and where
      // it used to be (in the case of a dynamic reallocation)
      int proc = getPatchwiseProcessorAssignment(patch);
      int oldproc = getOldProcessorAssignment(NULL, patch, 0);

      // we also need to see if the output processor for patch is this proc,
      // in case it wouldn't otherwise have been in the neighborhood
      int outputproc = (proc / d_outputNthProc)*d_outputNthProc;

      if(proc == me || oldproc == me || outputproc == me) {
	Patch::selectType n;
	IntVector lowIndex, highIndex;
	patch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0),
				      Ghost::AroundCells, maxGhost, n,
				      lowIndex, highIndex);
	for(int i=0;i<(int)n.size();i++){
	  const Patch* neighbor = n[i]->getRealPatch();
	  if(d_neighbors.find(neighbor) == d_neighbors.end())
	    d_neighbors.insert(neighbor);
	}
      }
    }
  }
}

bool
LoadBalancerCommon::inNeighborhood(const PatchSubset* ps,
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
LoadBalancerCommon::inNeighborhood(const Patch* patch)
{
  if(d_neighbors.find(patch) != d_neighbors.end())
    return true;
  else
    return false;
}

void
LoadBalancerCommon::problemSetup(ProblemSpecP& pspec, SimulationStateP& state)
{
  d_sharedState = state;
  d_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  string dynamicAlgo;
  double interval = 0;
  double cellFactor = .1;
  int timestepInterval = 0;
  d_outputNthProc = 1;
  double threshold = 0.0;
  
  if (p != 0) {
    p->getWithDefault("outputNthProc", d_outputNthProc, 1);
    if(!p->get("timestepInterval", timestepInterval))
      timestepInterval = 0;
    if (timestepInterval != 0 && !p->get("interval", interval))
      interval = 0.0; // default
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "static");
    p->getWithDefault("cellFactor", cellFactor, .1);
    p->getWithDefault("gainThreshold", threshold, 0.0);
  }

  setDynamicAlgorithm(dynamicAlgo, interval, timestepInterval, cellFactor, threshold);
}
