#include <Packages/Uintah/CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Util/FancyAssert.h>

#include <iostream> // debug only

using namespace Uintah;

using std::cerr;

#define DAV_DEBUG 0

RoundRobinLoadBalancer::RoundRobinLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld)
{
}

RoundRobinLoadBalancer::~RoundRobinLoadBalancer()
{
}

int
RoundRobinLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  int proc = patch->getID()%d_myworld->size();
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}

