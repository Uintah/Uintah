/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

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
  long long numProcs = d_myworld->size();
  const Patch* realPatch = patch->getRealPatch();
  int proc = (realPatch->getLevelIndex()*numProcs)/(long long)realPatch->getLevel()->numPatches();
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}
