
// $Id$

#include <Uintah/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>

using namespace Uintah;
using namespace std;

SingleProcessorLoadBalancer::SingleProcessorLoadBalancer( const ProcessorGroup* myworld )
   : UintahParallelComponent( myworld )
{
}

SingleProcessorLoadBalancer::~SingleProcessorLoadBalancer()
{
}

void SingleProcessorLoadBalancer::assignResources(TaskGraph& graph,
						  const ProcessorGroup*)
{
   int ntasks = graph.getNumTasks();
   for(int i=0;i<ntasks;i++)
      graph.getTask(i)->assignResource(0);
}

//
// $Log$
// Revision 1.1  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

