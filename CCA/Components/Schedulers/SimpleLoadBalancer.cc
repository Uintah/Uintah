

#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/DetailedTask.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

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
  int maxThreads = Parallel::getMaxThreads();
  int nTasks = graph.numTasks();
  int numProcs = group->size();

  // If there are less patches than threads, "divBy" will distribute
  // the work to both processors.
  int divBy = min( maxThreads, numProcs - 1 );
  // If there is only one processor, we need divBy to be 1.
  divBy = max( divBy, 1 );

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);
    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);
      int idx = (patch->getID() / divBy) % numProcs;
      task->assignResource(idx);
      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = (p->getID()/divBy)%numProcs;
	if(pidx != idx){
	  cerr << "WARNING: inconsistent task assignment in SimpleLoadBalancer\n";
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

int SimpleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
							const ProcessorGroup* group)
{
   return patch->getID()%group->size();
}
