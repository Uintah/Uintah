
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Semaphore.h>

#include <Uintah/Grid/Task.h>

#include <vector>
#include <stack>

namespace Uintah {

class ThreadPool;

using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;
using SCICore::Thread::Semaphore;

using std::vector;
using std::stack;

class Worker : public Runnable { 

public:
  
  Worker( ThreadPool * parent, int id, Semaphore * ready );

  void assignTask( Task * task, const ProcessorGroup * pg );

  virtual void run();

private:
  Semaphore            * d_ready;
  int                    d_id;
  ThreadPool           * d_parent;
  Task                 * d_task;
  const ProcessorGroup * d_pg; // PG of current d_task
};

class ThreadPool {

public:
  ThreadPool( int numWorkers );

  // Returns the number of available threads (0 if none available).
  int available();

  // Worker calls this to let the ThreadPool know that it is done
  // running its task...
  void done( int id, Task * task );

  // Returns a list of the tasks that have been completed.
  // The list is subsequently cleared and only updated with new tasks
  // that finish from the point of this call.
  void getFinishedTasks( vector<Task *> & finishedTasks );

  void assignThread( Task * task, const ProcessorGroup * pg );

private:

  int               d_numWorkers;
  int               d_numBusy;

  // List of tasks that workers have finished
  vector<Task *>    d_finishedTasks;

  // All workers and the threadPool can only access the workerQueue serially
  Semaphore       * d_workerQueueSem;

  // Used to unblock a worker thread after it has a task assigned to it.
  Semaphore      ** d_workerReadySems; 

  // An array of Worker Threads.
  Worker         ** d_workers;

  // A stack of the ids of available (idle) threads
  stack<int>        d_availableThreads;
};

} // end namespace Uintah
