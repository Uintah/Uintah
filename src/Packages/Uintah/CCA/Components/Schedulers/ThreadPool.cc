
#include <Packages/Uintah/CCA/Components/Schedulers/ThreadPool.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Time.h>
#include <Core/Util/Assert.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
#include <stdio.h>

using std::cerr;

using namespace Uintah;

using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
Mutex cerrLock( "cerr lock" );

#define DAV_DEBUG 0

/////////////// Worker //////////////////

Worker::Worker( ThreadPool * parent, int id, Mutex * ready ) : 
  d_ready( ready ), d_id( id ), d_parent( parent ), d_task( 0 ), d_pg( 0 )
{}

void
Worker::assignTask( Task * task, const ProcessorGroup * pg )
{
#if DAV_DEBUG
  cerrLock.lock();
  cerr << "Worker " << d_id << "- assignTask:   " << *task << "\n";
  cerrLock.unlock();
#endif
  ASSERT( !d_pg && !d_task );
  d_pg = pg;
  d_task = task;
}

void
Worker::run()
{
   for(;;){
    d_ready->lock();

    double beginTime = Time::currentSeconds();
#if DAV_DEBUG
    cerrLock.lock(); cerr << "Worker " << d_id << " running: " 
			  << *d_task << "\n"; cerrLock.unlock();
#endif

    ASSERT( d_task != 0 );

    Task * task = d_task;

#if 0
    d_task->doit( d_pg );
#else
    NOT_FINISHED("new task stuff");
#endif

#if DAV_DEBUG
    cerrLock.lock();cerr<<"Done:    " <<*d_task<< "\n";cerrLock.unlock();
#endif

    d_task = 0;
    d_pg = 0;

    double endTime = Time::currentSeconds();

    d_parent->done( d_id, task, endTime - beginTime );
  }
}

/////////////// ThreadPool //////////////////

ThreadPool::ThreadPool( int numWorkers ) :
  d_numWorkers( numWorkers ),
  d_numBusy( 0 )
{
  d_beginTime = Time::currentSeconds();

  d_timeUsed = scinew double[ numWorkers ];

  // Only one thing (worker thread or threadPool itself) can be
  // modifying the workerQueue (actually a stack) at a time.
  d_workerQueueLock = scinew Mutex( "ThreadPool Worker Queue Lock" );

  d_workerReadyLocks = scinew Mutex*[ numWorkers ];

  d_workers = scinew Worker*[ numWorkers ];

  for( int i = 0; i < numWorkers; i++ ){

    d_timeUsed[ i ] = 0.0;

    d_availableThreads.push( i );

    d_workerReadyLocks[ i ] = scinew Mutex( "Worker Ready Lock" );
    // None of the workers are allowed to run until the ThreadPool
    // tells them to run... therefore they must be locked initially.
    d_workerReadyLocks[ i ]->lock();

    Worker * worker = scinew Worker( this, i, d_workerReadyLocks[ i ] );

    char name[1024];
    sprintf( name, "Worker Thread %d", i );

    scinew Thread( worker, name );

    d_workers[ i ] = worker;
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
ThreadPool::assignThread( Task * task, const ProcessorGroup * pg )
{
  d_workerQueueLock->lock();

#if DAV_DEBUG
  cerr << "ThreadPool assignThread\n";
#endif

  if( d_availableThreads.empty() ){
    throw( InternalError( "All threads busy..." ) );
  }
  
  Worker * worker;
  int      id = d_availableThreads.top();

  d_availableThreads.pop();

  worker = d_workers[ id ];

  d_numBusy++;
  worker->assignTask( task, pg );

  d_workerReadyLocks[ id ]->unlock();

  d_workerQueueLock->unlock();
}

void
ThreadPool::done( int id, Task * task, double timeUsed )
{
  d_workerQueueLock->lock();

#if DAV_DEBUG
  cerrLock.lock();
  cerr << "Worker " << id << " finished: " << *task << "\n";
  cerrLock.unlock();
#endif

  d_finishedTasks.push_back( task );

  d_timeUsed[ id ] += timeUsed;

  d_availableThreads.push( id );
  d_numBusy--;

  d_workerQueueLock->unlock();
}

void
ThreadPool::getFinishedTasks( vector<Task *> & finishedTasks )
{
  d_workerQueueLock->lock();
  finishedTasks = d_finishedTasks;
  d_finishedTasks.clear();
  d_workerQueueLock->unlock();
}

