
// $Id$

#include <Uintah/Components/Schedulers/SimpleLoadBalancer.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Grid/Patch.h>

using namespace Uintah;

SimpleLoadBalancer::SimpleLoadBalancer(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

SimpleLoadBalancer::~SimpleLoadBalancer()
{
}

void SimpleLoadBalancer::assignResources(TaskGraph& graph,
					     const ProcessorGroup* group)
{
   int ntasks = graph.getNumTasks();
   int nump = group->size();
   for(int i=0;i<ntasks;i++){
      Task* task = graph.getTask(i);
      if(task->getPatch()){
	 // STILL THE SAME AS ROUNDROBIN - Steve
	 task->assignResource(task->getPatch()->getID()%nump);
      } else {
	 task->assignResource(0);
      }
   }
}

int SimpleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
							const ProcessorGroup* group)
{
   return patch->getID()%group->size();
}

//
// $Log$
// Revision 1.1  2000/09/20 16:00:28  sparker
// Added external interface to LoadBalancer (for per-processor tasks)
// Added message logging functionality. Put the tag <MessageLog/> in
//    the ups file to enable
//
//

