
#include <Uintah/Components/Schedulers/ThreadPool.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/Assert.h>

#include <iostream>

using std::cerr;
using namespace Uintah;
using SCICore::Exceptions::InternalError;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
Semaphore * cerrSem = scinew Semaphore( "cerr sema", 1 );

#define DAV_DEBUG 0

/////////////// Worker //////////////////

Worker::Worker( ThreadPool * parent, int id, Semaphore * ready ) : 
  d_task( 0 ), d_pg( 0 ), d_id( id ),  d_parent( parent ), d_ready( ready )
{}

void
Worker::assignTask( Task * task, const ProcessorGroup * pg )
{
#if DAV_DEBUG
  cerrSem->down();
  cerr << "Worker " << d_id << "- assignTask:   " << *task << "\n";
  cerrSem->up();
#endif
  ASSERT( !d_pg && !d_task );
  d_pg = pg;
  d_task = task;
}

void
Worker::run()
{
  while( 1 ){
    d_ready->down();

#if DAV_DEBUG
    cerrSem->down(); cerr << "Worker " << d_id << " running: " 
			  << *d_task << "\n"; cerrSem->up();
#endif

    ASSERT( d_task != 0 );

    Task * task = d_task;

    d_task->doit( d_pg );

#if DAV_DEBUG
    cerrSem->down(); cerr << "Done:    " << *d_task << "\n"; cerrSem->up();
#endif

    d_task = 0;
    d_pg = 0;
    d_parent->done( d_id, task );
  }
}

/////////////// ThreadPool //////////////////

ThreadPool::ThreadPool( int numWorkers ) :
  d_numWorkers( numWorkers ),
  d_numBusy( 0 )
{
  // Only one thing (worker thread or threadPool itself) can be
  // modifying the workerQueue (actually a stack) at a time.
  d_workerQueueSem = scinew Semaphore( "ThreadPool Worker Queue Sem", 1 );

  d_workerReadySems = scinew Semaphore*[ numWorkers ];

  d_workers = scinew Worker*[ numWorkers ];

  for( int i = 0; i < numWorkers; i++ ){

    char name[1024];

    d_availableThreads.push( i );

    d_workerReadySems[ i ] = scinew Semaphore( "Worker Ready Sem", 0 );

    Worker * worker = scinew Worker( this, i, d_workerReadySems[ i ] );

    sprintf( name, "Worker Thread %d", i );
    scinew Thread( worker, name );

    d_workers[ i ] = worker;
  }
}

int
ThreadPool::available()
{
  d_workerQueueSem->down();
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

  d_workerQueueSem->up();

  return numAvail;
}

void
ThreadPool::assignThread( Task * task, const ProcessorGroup * pg )
{
  d_workerQueueSem->down();

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

  d_workerReadySems[ id ]->up();

  d_workerQueueSem->up();
}

void
ThreadPool::done( int id, Task * task )
{
  d_workerQueueSem->down();

#if DAV_DEBUG
  cerrSem->down();
  cerr << "Worker " << id << " finished: " << *task << "\n";
  cerrSem->up();
#endif

  d_finishedTasks.push_back( task );

  d_availableThreads.push( id );
  d_numBusy--;

  d_workerQueueSem->up();
}

void
ThreadPool::getFinishedTasks( vector<Task *> & finishedTasks )
{
  d_workerQueueSem->down();
  finishedTasks = d_finishedTasks;
  d_finishedTasks.clear();
  d_workerQueueSem->up();
}
