

#include <Packages/Uintah/CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Level.h>

using namespace Uintah;
using namespace std;

SingleProcessorLoadBalancer::SingleProcessorLoadBalancer( const ProcessorGroup* myworld )
   : LoadBalancerCommon( myworld )
{
}

SingleProcessorLoadBalancer::~SingleProcessorLoadBalancer()
{
}

void SingleProcessorLoadBalancer::assignResources(DetailedTasks& graph)
{
  int ntasks = graph.numTasks();
  for(int i=0;i<ntasks;i++)
    graph.getTask(i)->assignResource(0);
}

int SingleProcessorLoadBalancer::getPatchwiseProcessorAssignment(const Patch*)
{
   return 0;
}

const PatchSet*
SingleProcessorLoadBalancer::createPerProcessorPatchSet(const LevelP& level)
{
  return level->allPatches();
}


void
SingleProcessorLoadBalancer::createNeighborhood(const GridP&)
{
  // Nothing to do
}

bool
SingleProcessorLoadBalancer::inNeighborhood(const PatchSubset*,
					    const MaterialSubset*)
{
  return true;
}

bool
SingleProcessorLoadBalancer::inNeighborhood(const Patch*)
{
  return true;
}
