

#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

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

void RoundRobinLoadBalancer::assignResources(TaskGraph& graph,
					     const ProcessorGroup* group)
{
   int maxThreads = Parallel::getMaxThreads();
   int nTasks = graph.getNumTasks();
   int numProcs = group->size();

   for(int i=0;i<nTasks;i++){
      Task* task = graph.getTask(i);
      if(task->getPatch()){
	 // If there are less patches than threads, "divBy" will distribute
	 // the work to both processors.
	 int divBy = min( maxThreads, numProcs - 1 );
	 // If there is only one processor, we need divBy to be 1.
	 divBy = max( divBy, 1 );
	 task->assignResource( (task->getPatch()->getID() / divBy) % numProcs);
      } else {
	if( Parallel::usingMPI() && task->isReductionTask() ){
	  task->assignResource( Parallel::getRootProcessorGroup()->myrank() );
	} else {
#if DAV_DEBUG
	  cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
	  task->assignResource(0);
	}
      }
   }
}

int RoundRobinLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
							    const ProcessorGroup* group)
{
   return patch->getID()%group->size();
}


