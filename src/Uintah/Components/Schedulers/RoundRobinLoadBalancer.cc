
// $Id$

#include <Uintah/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Grid/Patch.h>

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

//
// $Log$
// Revision 1.4.2.1  2000/10/26 10:05:56  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/09/28 22:18:35  dav
// assignResource updates
//
// Revision 1.4  2000/09/27 02:12:57  dav
// Mixed Model Changes.
//
// Revision 1.3  2000/09/20 16:00:28  sparker
// Added external interface to LoadBalancer (for per-processor tasks)
// Added message logging functionality. Put the tag <MessageLog/> in
//    the ups file to enable
//
// Revision 1.2  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.1  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

