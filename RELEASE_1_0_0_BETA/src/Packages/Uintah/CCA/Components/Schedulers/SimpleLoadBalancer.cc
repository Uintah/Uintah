

#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

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


