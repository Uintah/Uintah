

#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>

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

int SingleProcessorLoadBalancer::getPatchwiseProcessorAssignment(const Patch*,
								 const ProcessorGroup*)
{
   return 0;
}


