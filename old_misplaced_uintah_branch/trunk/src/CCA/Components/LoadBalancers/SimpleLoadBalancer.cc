#include <CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

#include <SCIRun/Core/Util/FancyAssert.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <SCIRun/Core/Thread/Mutex.h>

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
