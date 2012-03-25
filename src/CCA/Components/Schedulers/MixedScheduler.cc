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



#include <CCA/Components/Schedulers/MixedScheduler.h>

#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <set>
#include <map>
#include <algorithm>


using namespace Uintah;
using namespace SCIRun;

using namespace std;

static Mutex send_data( "MPIScheduler::send_data" );

#define DAV_DEBUG 0

static DebugStream dbg("MixedScheduler", false);

MixedScheduler::MixedScheduler(const ProcessorGroup* myworld, Output* oport)
   : MPIScheduler(myworld, oport), log(myworld, oport)
{
  int num_threads = Parallel::getNumThreads();
  d_threadPool = scinew ThreadPool( this, num_threads, num_threads );
}

MixedScheduler::~MixedScheduler()
{
  cout << "Destructor for MixedScheduler...\n";
  delete d_threadPool;
}

SchedulerP
MixedScheduler::createSubScheduler()
{
  MixedScheduler* newsched = scinew MixedScheduler(d_myworld, m_outPort);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

bool
MixedScheduler::useInternalDeps()
{
  return true;
}

void
MixedScheduler::wait_till_all_done()
{
  // Blocks until threadpool is empty.
  d_threadPool->all_done();
}

void
MixedScheduler::initiateTask( DetailedTask * task,
			      bool /* only_old_recvs */,
			      int  /* abort_point */,
                              int  /* iteration */ )
{
  d_threadPool->addTask( task );
}

void
MixedScheduler::initiateReduction( DetailedTask          * task )
{
  d_threadPool->addReductionTask( task );
}

void
MixedScheduler::problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& /* state */ )
{
  log.problemSetup( prob_spec );
}

