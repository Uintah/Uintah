
// $Id$

#include <Uintah/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>

using namespace Uintah;

RoundRobinLoadBalancer::RoundRobinLoadBalancer( int MpiRank, int MpiProcesses)
   : UintahParallelComponent( MpiRank, MpiProcesses )
{
}

RoundRobinLoadBalancer::~RoundRobinLoadBalancer()
{
}

void RoundRobinLoadBalancer::assignResources(TaskGraph& graph,
					     const vector<ProcessorGroup*>& resources)
{
   int ntasks = graph.getNumTasks();
   int nump = (int)resources.size();
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
// Revision 1.1  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

