#include <CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <Core/Grid/Grid.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <SCIRun/Core/Util/FancyAssert.h>

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

