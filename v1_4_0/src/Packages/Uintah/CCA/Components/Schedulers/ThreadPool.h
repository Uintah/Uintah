#ifndef UINTAH_SCHEDULERS_THREADPOOL_H
#define UINTAH_SCHEDULERS_THREADPOOL_H

#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>

#include <vector>
#include <stack>

namespace Uintah {

class ThreadPool;

using namespace SCIRun;

using std::vector;
using std::stack;

class SendState; // from SendState.h
struct mpi_timing_info_s; // from MPIScheduler.h

class Worker : public Runnable { 

public:
  
  Worker( ThreadPool * parent, int id, Mutex * ready, Semaphore * do_work );

  void assignTask( const ProcessorGroup  * pg,
		   DetailedTask          * task,
		   mpi_timing_info_s     & mpi_info,
		   SendRecord            & sends,
		   SendState             & ss,
		   OnDemandDataWarehouse * dws[2],
		   const VarLabel        * reloc_label );

  virtual void run();

  void quit();

private:

  Semaphore            * do_work_;
  Mutex                * d_ready;
  int                    d_id;
  int                    proc_group_;
  ThreadPool           * d_parent;

  DetailedTask          * d_task;
  const ProcessorGroup  * d_pg; // PG of current d_task
  mpi_timing_info_s     * mpi_info_;
  SendRecord            * sends_;
  SendState             * ss_;
  OnDemandDataWarehouse * dws_[2];
  const VarLabel        * reloc_label_;

  bool                    quit_;
};

class ThreadPool {

public:

  // maxThreads:    total number of threads to create.
  // maxConcurrent: total allowed to run at same time.
  ThreadPool( int maxThreads, int maxConcurrent );

  ~ThreadPool();

  // Returns the number of available threads (0 if none available).
  int available();

  // Worker calls this to let the ThreadPool know that it is done
  // running its task...
  void done( int id, DetailedTask * task, double timeUsed );

  void assignThread( const ProcessorGroup  * pg,
		     DetailedTask          * task,
		     mpi_timing_info_s     & mpi_info,
		     SendRecord            & sends,
		     SendState             & ss,
		     OnDemandDataWarehouse * dws[2],
		     const VarLabel        * reloc_label );

  // Blocks until thread pool is empty.
  void all_done();

  // Returns the percent of time that has been used by threads
  // for executing tasks.
  double getUtilization();

private:

  double            d_beginTime;
  int               d_numWorkers;
  volatile int      d_numBusy;


  // List of the amount of time used by each thread for processing tasks.
  double          * d_timeUsed;

  // All workers and the threadPool can only access the workerQueue serially
  Mutex           * d_workerQueueLock;

  // Used to unblock a worker thread after it has a task assigned to it.
  Mutex          ** d_workerReadyLocks; 

  // An array of Worker Threads.
  Worker         ** d_workers;

  Semaphore       * num_threads_;

  // A stack of the ids of available (idle) threads
  stack<int>        d_availableThreads;
};

} // End namespace Uintah

#endif
