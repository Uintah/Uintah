
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

SimpleLoadBalancer::SimpleLoadBalancer(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

SimpleLoadBalancer::~SimpleLoadBalancer()
{
}

void SimpleLoadBalancer::assignResources(DetailedTasks& graph,
					 const ProcessorGroup* group)
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

      int idx = getPatchwiseProcessorAssignment(patch, group);
      ASSERTRANGE(idx, 0, group->size());

      task->assignResource(idx);

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "1) Task " << *(task->getTask()) << " put on resource "
		   << idx << "\n";
	cerrLock.unlock();
      }

      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = getPatchwiseProcessorAssignment(p, group);
	ASSERTRANGE(pidx, 0, group->size());
	if(pidx != idx){
	  cerrLock.lock();
	  cerr << "WARNING: inconsistent task (" << task->getTask()->getName() 
	       << ") assignment (" << pidx << ", " << idx 
	       << ") in SimpleLoadBalancer\n";
	  cerrLock.unlock();
	}
      }
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
	task->assignResource( Parallel::getRootProcessorGroup()->myrank() );

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "  Resource (for no patch task) is : " 
		     << Parallel::getRootProcessorGroup()->myrank() << "\n";
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
}

int
SimpleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
						    const ProcessorGroup* group)
{
  int numProcs = group->size();
  int proc = (patch->getLevelIndex()*numProcs)/patch->getLevel()->numPatches();
  ASSERTRANGE(proc, 0, group->size());
  return proc;
}

const PatchSet*
SimpleLoadBalancer::createPerProcessorPatchSet(const LevelP& level,
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
SimpleLoadBalancer::createNeighborhood(const GridP& grid,
				       const ProcessorGroup* group,
				       const Scheduler* /*sch*/)
{
  int me = group->myrank();
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
      if(getPatchwiseProcessorAssignment(patch, group) == me){
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
SimpleLoadBalancer::inNeighborhood(const PatchSubset* ps,
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
SimpleLoadBalancer::inNeighborhood(const Patch* patch)
{
  if(d_neighbors.find(patch) != d_neighbors.end())
    return true;
  else
    return false;
}
