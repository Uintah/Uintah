

#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;
using namespace std;

SingleProcessorLoadBalancer::SingleProcessorLoadBalancer( const ProcessorGroup* myworld )
   : UintahParallelComponent( myworld )
{
}

SingleProcessorLoadBalancer::~SingleProcessorLoadBalancer()
{
}

void SingleProcessorLoadBalancer::assignResources(DetailedTasks& graph,
						  const ProcessorGroup*)
{
  int ntasks = graph.numTasks();
  for(int i=0;i<ntasks;i++)
    graph.getTask(i)->assignResource(0);
}

int SingleProcessorLoadBalancer::getPatchwiseProcessorAssignment(const Patch*,
								 const ProcessorGroup*)
{
   return 0;
}


