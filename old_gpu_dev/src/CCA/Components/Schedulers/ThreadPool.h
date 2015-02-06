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


#ifndef UINTAH_SCHEDULERS_THREADPOOL_H
#define UINTAH_SCHEDULERS_THREADPOOL_H

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/CommRecMPI.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/RecursiveMutex.h>
#include <Core/Thread/Semaphore.h>

#include <vector>
#include <stack>

namespace Uintah {

class Receiver;
class Worker;
class MPIReducer;
class MixedScheduler;
class DetailedTask;
  
using namespace SCIRun;

class SendState; // from SendState.h
struct mpi_timing_info_s; // from MPIScheduler.h

class ThreadPool {

public:
  typedef std::map< int, std::list<Receiver*>* > ReceiverPriorityQueue;
  typedef std::pair< std::list<Receiver*>*, std::list<Receiver*>::iterator >
  ReceiverPriorityQueueItem;

public:
  // maxThreads:    total number of threads to create.
  // maxConcurrent: total allowed to run at same time.
  ThreadPool( MixedScheduler* scheduler, int maxThreads, int maxConcurrent );

  ~ThreadPool();

  void addTask( DetailedTask          * task );
  void addReductionTask( DetailedTask          * task );

  // Receiver calls this to start a worker on a task that has received all
  // needed data.
  void launchTask( DetailedTask          * task );
    
  // Worker calls this to let the ThreadPool know that it is done
  // running its task...
  void done( int id, DetailedTask * task, double timeUsed );
  void doneReduce(DetailedTask * task);

  // Returns the number of available threads (0 if none available).
  int available();
  
  // Blocks until thread pool is empty.
  void all_done();

  // Returns the percent of time that has been used by threads
  // for executing tasks.
  double getUtilization();

  // called by a Receiver when one of its byte count has changed
  bool reprioritize(Receiver* receiver,
		    ReceiverPriorityQueueItem& item /* gets changed */,
		    bool tasksAdded, int oldBytes);

  MixedScheduler* getScheduler()
  { return d_scheduler; }  
private:
  MixedScheduler* d_scheduler;
  
  double            d_beginTime;
  int               d_maxWorkers;
  int               d_maxReceivers;
  volatile int      d_numBusy; // # of workers busy running a task
  volatile int      d_numWaiting; // # of tasks scheduled but not running
  volatile int      d_numReduces; 
  volatile int      d_numReduced; 

  // List of the amount of time used by each thread for processing tasks.
  double          * d_timeUsed;

  // All workers and the threadPool can only access the workerQueue serially
  Mutex           * d_workerQueueLock;

  // Used to unblock a worker thread after it has a task assigned to it.
  Mutex          ** d_workerReadyLocks; 

  // An array of Worker and Receiver Threads.
  Worker         ** d_workers;
  Semaphore       * num_workers_;
  
  Receiver       ** d_receivers;

  MPIReducer      * d_reducer;
 
  // A stack of the ids of available (idle) worker threads
  std::stack<int>      d_availableWorkers;

  // prioritize receivers based on how my bytes of unfinished batches they
  // are waiting on.
  ReceiverPriorityQueue     d_receiverQueue;
  Mutex                     d_receiverQueueLock;
  int                       d_numReceiversAddingTasks;
  int                       d_maxBytesWhileAddingTasks; // see addTask

  Thread*                   d_waitingForReprioritizingThread;
  Thread*                   d_waitingTilDoneThread;
};  
  
class Worker : public Runnable { 

public:
  
  Worker( ThreadPool * parent, int id, Mutex * ready );

  void assignTask( DetailedTask* task );

  virtual void run();

  void quit();

private:

  Mutex                * d_ready;
  int                    d_id;
  ThreadPool           * d_parent;
  int                    proc_group_;

  DetailedTask          * d_task;
  bool                    quit_;
};

class Receiver : public Runnable { 

public:
  
  Receiver( ThreadPool * parent, int id );

  void assignTask( DetailedTask* task );

  virtual void run();

  void quit();

  int getUnfinishedBytes() const
  { return recvs_.getUnfinishedBytes(); }

  void setPriorityItem(ThreadPool::ReceiverPriorityQueueItem& item)
  { priorityItem_ = item; }

  // was assigned new tasks that haven't been added to the awaiting list yet
  bool hasNewTasks()
  { return !newTasks_.empty(); }
private:
  // add new tasks to the awaiting list
  bool addAwaitingTasks(); // returns true if any tasks were added
  
  class AwaitingTask {
  public:
    AwaitingTask(DetailedTask* task,
		 std::list<DependencyBatch*> outstandingExtRecvs, int threadID);
    DetailedTask* getTask()
    { return task_; }
    bool isReady();
  private:
    DetailedTask* task_;
    // external receives outstanding
    std::list<DependencyBatch*> outstandingExtRecvs_;
  };

  int                    d_id;
  ThreadPool           * d_parent;
  int                    proc_group_;
  const ProcessorGroup*  pg_;
  Mutex                  d_lock;  
  
  CommRecMPI                  recvs_;
  std::vector<AwaitingTask*>  awaitingTasks_;
  std::queue<int>             availTaskSlots_;
  std::queue<DetailedTask*>   newTasks_;

  // awaitingTasks_ indices of tasks that have received data they have
  // personally posted requests for (but may still be waiting for others'
  // requests -- outstandingExtRecvs_).
  std::list<int>    semiReadyTasks_;  

  ThreadPool::ReceiverPriorityQueueItem priorityItem_;
  
  bool                    quit_;
};

class MPIReducer : public Runnable { 

public:
  
  MPIReducer(ThreadPool * parent);

  void assignTask( DetailedTask* task );

  virtual void run();

  void quit();
  
private:
  ThreadPool*                d_parent;
  std::queue<DetailedTask*>  tasks_;
  Mutex                      lock_;
  bool                       paused_;
  bool                       quit_;
  int                        proc_group_;
};
  

} // End namespace Uintah

#endif
