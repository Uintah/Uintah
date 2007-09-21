
#include <CCA/Components/Schedulers/MixedScheduler.h>

#include <SCIRun/Core/Thread/Time.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <SCIRun/Core/Util/FancyAssert.h>
#include <SCIRun/Core/Malloc/Allocator.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <sgi_stl_warnings_off.h>
#include <set>
#include <map>
#include <algorithm>
#include <sgi_stl_warnings_on.h>


using namespace Uintah;
using namespace SCIRun;

using std::cerr;
using std::find;

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

