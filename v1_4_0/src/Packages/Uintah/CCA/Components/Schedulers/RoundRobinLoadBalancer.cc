

#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

#include <iostream> // debug only

using namespace Uintah;

using std::cerr;

#define DAV_DEBUG 0

RoundRobinLoadBalancer::RoundRobinLoadBalancer(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

RoundRobinLoadBalancer::~RoundRobinLoadBalancer()
{
}

void RoundRobinLoadBalancer::assignResources(DetailedTasks& graph,
					     const ProcessorGroup* group)
{
  int nTasks = graph.numTasks();
  int numProcs = group->size();

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);
    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);
      int idx = patch->getID() % numProcs;
      task->assignResource(idx);
      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = p->getID()%numProcs;
	if(pidx != idx){
	  cerr << "WARNING: inconsistent task assignment in RoundRobinLoadBalancer\n";
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
RoundRobinLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
							const ProcessorGroup* group)
{
   return patch->getID()%group->size();
}

const PatchSet*
RoundRobinLoadBalancer::createPerProcessorPatchSet(const LevelP& level,
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
  return patches;
}


void
RoundRobinLoadBalancer::createNeighborhood(const Level* level,
					   const ProcessorGroup* group)
{
  int me = group->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  neighbors.clear();
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    if(getPatchwiseProcessorAssignment(patch, group) == me){
      Level::selectType n;
      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::CellBased, Ghost::AroundCells,
				    maxGhost, n, lowIndex, highIndex);
      for(int i=0;i<(int)n.size();i++){
	const Patch* neighbor = n[i];
	if(neighbors.find(neighbor) == neighbors.end())
	  neighbors.insert(neighbor);
      }
    }
  }
}

bool
RoundRobinLoadBalancer::inNeighborhood(const PatchSubset* ps,
				       const MaterialSubset*)
{
  for(int i=0;i<ps->size();i++){
    const Patch* patch = ps->get(i);
    if(neighbors.find(patch) != neighbors.end())
      return true;
  }
  return false;
}

bool
RoundRobinLoadBalancer::inNeighborhood(const Patch* patch)
{
  if(neighbors.find(patch) != neighbors.end())
    return true;
  else
    return false;
}
