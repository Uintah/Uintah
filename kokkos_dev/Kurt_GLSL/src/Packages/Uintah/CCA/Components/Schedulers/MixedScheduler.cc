
#include <Packages/Uintah/CCA/Components/Schedulers/MixedScheduler.h>

#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <sgi_stl_warnings_off.h>
#include <set>
#include <map>
#include <algorithm>
#include <sgi_stl_warnings_on.h>


using namespace Uintah;
using namespace SCIRun;

using std::cerr;
using std::find;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
//extern Mutex cerrLock;
//extern DebugStream mixedDebug;

static Mutex send_data( "MPIScheduler::send_data" );

#define DAV_DEBUG 0

static DebugStream dbg("MixedScheduler", false);

MixedScheduler::MixedScheduler(const ProcessorGroup* myworld, Output* oport)
   : MPIScheduler(myworld, oport), log(myworld, oport)
{
  int num_threads = Parallel::getMaxThreads();
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
  MixedScheduler* newsched = new MixedScheduler(d_myworld, m_outPort);
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
MixedScheduler::initiateTask( DetailedTask          * task,
			      bool /* only_old_recvs */,
			      int /* abort_point */)
{
  d_threadPool->addTask( task );
}

void
MixedScheduler::initiateReduction( DetailedTask          * task )
{
  d_threadPool->addReductionTask( task );
}

void
MixedScheduler::problemSetup( const ProblemSpecP& prob_spec )
{
  log.problemSetup( prob_spec );
}

