
// $Id$

#include <Uintah/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Grid/Patch.h>

using namespace Uintah;

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
   int ntasks = graph.getNumTasks();
   int nump = group->size();
   for(int i=0;i<ntasks;i++){
      Task* task = graph.getTask(i);
      if(task->getPatch()){
	 task->assignResource(task->getPatch()->getID()%nump);
      } else {
	 task->assignResource(0);
      }
   }
}

//
// $Log$
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

