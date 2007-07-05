#include <Packages/Uintah/CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex cerrLock;
extern SCIRun::DebugStream lbDebug;

SimpleLoadBalancer::SimpleLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld)
{
}

SimpleLoadBalancer::~SimpleLoadBalancer()
{
}

int
SimpleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  int numProcs = d_myworld->size();
  const Patch* realPatch = patch->getRealPatch();
  int proc = (realPatch->getLevelIndex()*numProcs)/realPatch->getLevel()->numPatches();
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}
