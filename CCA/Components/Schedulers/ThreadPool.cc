
#include <Packages/Uintah/CCA/Components/Schedulers/ThreadPool.h>

#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Time.h>
#include <Core/Util/Assert.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <sstream>
#include <stdio.h>

using std::cerr;

using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

#define DAV_DEBUG 1

#ifdef USE_TAU_PROFILING

map<string,int> taskname_to_id_map;
int unique_id = 9999;

int
create_tau_mapping( const string & taskname, const PatchSubset * patches )
{
  string full_name = taskname;
 
  if( patches ) {
    for(int i=0;i<patches->size();i++) {

      ostringstream patch_num;
      patch_num << patches->get(i)->getID();

      full_name = full_name + "-" + patch_num.str();
    }
  }

  map<string,int>::iterator iter = taskname_to_id_map.find( full_name );
  if( iter != taskname_to_id_map.end() )
    {
      return (*iter).second;
    }
  else
    {
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "creating TAU mapping for: " << full_name << "\n";
	cerrLock.unlock();
      }
      TAU_MAPPING_CREATE( full_name, "[MPIScheduler::execute()]",
			  (TauGroup_t) unique_id, full_name.c_str(), 0 );
      taskname_to_id_map[ full_name ] = unique_id;
      unique_id++;
      return (unique_id - 1);
    }
}
#endif // USE_TAU_PROFILING

/////////////// Worker //////////////////

Worker::Worker( ThreadPool * parent, int id, 
		Mutex * ready, Semaphore * do_work ) : 
  d_ready( ready ), d_id( id ), d_parent( parent ), d_task( 0 ), d_pg( 0 ),
  do_work_( do_work ), mpi_info_( 0 ), sends_( 0 ), ss_( 0 ), quit_( false )
{
  proc_group_ = Parallel::getRootProcessorGroup()->myrank();
}

void
Worker::quit()
{
  quit_ = true;
  d_ready->unlock();  
}

void
Worker::assignTask( const ProcessorGroup  * pg,
		    DetailedTask          * task,
		    mpi_timing_info_s     & mpi_info,
		    SendRecord            & sends,
		    SendState             & ss,
		    OnDemandDataWarehouse * dws[2],
		    const VarLabel        * reloc_label )
{
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << proc_group_ << "-" << d_id 
	       << ": assignTask:   " << *task << "\n";
    cerrLock.unlock();
  }
#endif
  ASSERT( !d_pg && !d_task );

  d_task = task;
  d_pg = pg;
  mpi_info_ = &mpi_info;
  sends_ = &sends;
  ss_ = &ss;
  dws_[ Task::NewDW ] = dws[ Task::NewDW ];
  dws_[ Task::OldDW ] = dws[ Task::OldDW ];
  reloc_label_ = reloc_label;
}

void
Worker::run()
{
  cout << "here\n";

  TAU_REGISTER_THREAD();
  TAU_PROFILE("Worker_run()", "void ()", TAU_DEFAULT);
  TAU_PROFILE_TIMER(doittimer, "doit Task", "[Worker::run()]", TAU_DEFAULT);

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << proc_group_ << "-" << d_id 
	       << ", Begin run() -- PID is " << getpid() << "\n";
    cerrLock.unlock();
  }

  for(;;){

    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " waiting for a task\n";
      cerrLock.unlock();
    }

    d_ready->lock();

    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " unlocked\n";
      cerrLock.unlock();
    }

    if( quit_ ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " quitting\n";
      cerrLock.unlock();
      TAU_PROFILE_EXIT( "thread quitting" );
      return;
    }

    double beginTime = Time::currentSeconds();
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " running: " << *d_task << "\n";
      cerrLock.unlock();
    }
#endif

    ASSERT( d_task != 0 );

    DetailedTask * task = d_task;

    // An unlimited number of threads may post their MPI recv requests.
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
				  << " calling mpi recv\n";
      cerrLock.unlock();
    }
#endif
    MPIScheduler::recvMPIData( d_pg, task, *mpi_info_, dws_ );
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
				  << " recved all mpi data\n";
      cerrLock.unlock();
    }
#endif

    // However, only N threads are allowed to actually "do_work" at
    // the same time.  This semaphore assures that.
    do_work_->down();

#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
				  << " calling doit for detailed task: " 
				  << task << " which has task: " 
				  << task->getTask() << "\n";
      cerrLock.unlock();
    }
#endif
    
    try {

#ifdef USE_TAU_PROFILING
      int id;
      const PatchSubset* patches = task->getPatches();
      id = create_tau_mapping( task->getTask()->getName(), patches );
#endif

  // Should this be here?
  TAU_PROFILE_TIMER(doittimer, "Task execution", "[Worker::run()] ",
		    TAU_USER); 
  TAU_MAPPING_OBJECT(tautimer)
  TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);  // EXTERNAL ASSOCIATION
  TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)
  TAU_PROFILE_START(doittimer);
  TAU_MAPPING_PROFILE_START(doitprofiler,0);

      task->doit( d_pg, dws_[ Task::OldDW ], dws_[ Task::NewDW ] );

  TAU_MAPPING_PROFILE_STOP(0);
  TAU_PROFILE_STOP(doittimer);

  TAU_DB_DUMP();


#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
				<< " done with doit, calling done()\n";
    mixedDebug << "detailed task is " << task << " with task "
	       << task->getTask() << "\n";
    cerrLock.unlock();
  }
#endif
      task->done();

    } catch (Exception& e) {

      cerrLock.lock(); cerr << "Worker " << proc_group_ << "-" << d_id 
			    << ": Caught exception: " 
			    << e.message() << '\n';
      if(e.stackTrace())
	cerr << "Stack trace: " << e.stackTrace() << '\n';
      cerrLock.unlock();

      //      Thread::exitAll(1);
    } catch (std::exception e){

      cerrLock.lock();
      cerr << "Worker " << proc_group_ << "-" << d_id 
	   << ": Caught std exception: " << e.what() << '\n';
      cerrLock.unlock();
      //      Thread::exitAll(1);
    } catch(...){
      cerrLock.lock(); cerr << "Worker " << proc_group_ << "-" << d_id 
			    << ": Caught unknown exception\n";
      cerrLock.unlock();
      //      Thread::exitAll(1);
    }

#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
			    << " done with done()\n";
      cerrLock.unlock();
    }
#endif
    MPIScheduler::sendMPIData( d_pg, task, *mpi_info_, *sends_, 
			       *ss_, dws_, reloc_label_ );

#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug<<"Done with send mpi data: " <<*d_task<< "\n";
      cerrLock.unlock();
    }
#endif

    d_task       = 0;
    d_pg         = 0;
    dws_[ Task::OldDW ] = 0;
    dws_[ Task::NewDW ] = 0;
    reloc_label_ = 0;

    double endTime = Time::currentSeconds();

    do_work_->up();
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug<<" do_work upped\n";
      cerrLock.unlock();
    }
#endif
    d_parent->done( d_id, task, endTime - beginTime );
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << " told parent i am done\n";
      cerrLock.unlock();
    }
#endif
  }
}

/////////////// ThreadPool //////////////////

ThreadPool::ThreadPool( int maxThreads, int maxConcurrent ) :
  d_numWorkers( maxThreads ),
  d_numBusy( 0 )
{
  d_beginTime = Time::currentSeconds();

  d_timeUsed = scinew double[ maxThreads ];

  num_threads_ = scinew Semaphore( "number of threads", maxThreads );

  Semaphore * do_work = 
                 scinew Semaphore( "max running thread limit", maxConcurrent );

  // Only one thing (worker thread or threadPool itself) can be
  // modifying the workerQueue (actually a stack) at a time.
  d_workerQueueLock = scinew Mutex( "ThreadPool Worker Queue Lock" );

  d_workerReadyLocks = scinew Mutex*[ maxThreads ];

  d_workers = scinew Worker*[ maxThreads ];

  for( int i = 0; i < maxThreads; i++ ){

    d_timeUsed[ i ] = 0.0;

    d_availableThreads.push( i );

    d_workerReadyLocks[ i ] = scinew Mutex( "Worker Ready Lock" );
    // None of the workers are allowed to run until the ThreadPool
    // tells them to run... therefore they must be locked initially.
    d_workerReadyLocks[ i ]->lock();

    Worker * worker = scinew Worker( this, i,
				     d_workerReadyLocks[ i ], do_work );

    char name[1024];
    sprintf( name, "Worker %d-%d",
	     Parallel::getRootProcessorGroup()->myrank(), i );

    Thread * t = scinew Thread( worker, name );
    t->detach();

    d_workers[ i ] = worker;
  }
  cout << "Done creating worker threads\n";
}

ThreadPool::~ThreadPool()
{
  // Wait for all threads to be finished doing what they were doing.
  cout << "in destructor: Waiting for all threads to finish\n";
  while( available() < d_numWorkers )
    {
      usleep( 1000 );
    }

  cout << "Telling all threads to quit\n";
  for( int i = 0; i < d_numWorkers; i++ ){
    d_workers[ i ]->quit();
  }
}


double
ThreadPool::getUtilization()
{
  double time = Time::currentSeconds();
  double totalTimeUsed = 0;
  double maxTime = d_numWorkers * ( time - d_beginTime );

  d_workerQueueLock->lock();

  for( int i = 0; i < d_numWorkers; i++ ){
    totalTimeUsed += d_timeUsed[ i ];
  }

  d_workerQueueLock->unlock();

  return totalTimeUsed / maxTime;
}

int
ThreadPool::available()
{
  d_workerQueueLock->lock();
  int numAvail = d_numWorkers - d_numBusy;

#if DAV_DEBUG
  static int cnt = 0;
  static double avg = 0;
  static int minFree = 99999;
  if( numAvail < minFree ){
    minFree = numAvail;
  }
  cnt++;
  avg += d_numBusy;
  if( cnt % 50000 == 0 ){
    cerr << "ThreadPool: number of threads available: " 
	 << d_numWorkers - d_numBusy << ", min: " << minFree << "\n";
    cerr << "            avg busy: " << avg / cnt << "\n";
  }
#endif

  d_workerQueueLock->unlock();

  return numAvail;
}

void
ThreadPool::assignThread( const ProcessorGroup  * pg,
			  DetailedTask          * task,
			  mpi_timing_info_s     & mpi_info,
			  SendRecord            & sends,
			  SendState             & ss,
			  OnDemandDataWarehouse * dws[2],
			  const VarLabel        * reloc_label )
{

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "ThreadPool pre-assignThread\n";
    cerrLock.unlock();
  }
#endif

  num_threads_->down();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug<< "ThreadPool: thread available\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->lock();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "ThreadPool now-assignThread\n";
    cerrLock.unlock();
  }
#endif

  //  if( d_availableThreads.empty() ){
  //    throw( InternalError( "All threads busy..." ) );
  //  }
  
  Worker * worker;
  int      id = d_availableThreads.top();

  d_availableThreads.pop();

  worker = d_workers[ id ];

  d_numBusy++;
  worker->assignTask( pg, task, mpi_info, sends, ss, dws, reloc_label );

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Unlocking Worker Thread " << id << "\n";
    cerrLock.unlock();
  }
#endif

  d_workerReadyLocks[ id ]->unlock();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker Thread " << id << " unlocked\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->unlock();
}

void
ThreadPool::all_done()
{
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "waiting for all threads to finish\n";
    cerrLock.unlock();
  }

  int cnt = 0;

  // Dd: Is there a beter way to do this?
  while( d_numBusy > 0 ) {

    cnt++;
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << "all threads are not done yet\n";
      cerrLock.unlock();
    }
#endif
  }
  // if( mixedDebug.active() ) {
    cerrLock.lock();
    cerr << "all threads now done: d_numBusy is " << d_numBusy << "\n";
    cerrLock.unlock();
    //}
}

void
ThreadPool::done( int id, DetailedTask * task, double timeUsed )
{
  
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug<< "Worker " << id << " done.  getting queue lock\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->lock();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << id << " finished: " << *task << "\n";
    cerrLock.unlock();
  }
#endif

  d_timeUsed[ id ] += timeUsed;

  d_availableThreads.push( id );
  d_numBusy--;

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "numBusy: " << d_numBusy << "\n"; 
    cerrLock.unlock();
  }

  num_threads_->up();

  d_workerQueueLock->unlock();
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << id 
	       << " added back to the list of threads available\n";
    cerrLock.unlock();
  }
#endif
}


