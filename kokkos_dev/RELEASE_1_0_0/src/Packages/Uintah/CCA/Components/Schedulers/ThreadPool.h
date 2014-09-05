
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

#include <Packages/Uintah/Core/Grid/Task.h>

#include <vector>
#include <stack>

namespace Uintah {
class ThreadPool;

using namespace SCIRun;

using std::vector;
using std::stack;

class Worker : public Runnable { 

public:
  
  Worker( ThreadPool * parent, int id, Mutex * ready );

  void assignTask( Task * task, const ProcessorGroup * pg );

  virtual void run();

private:

  Mutex                * d_ready;
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
  void done( int id, Task * task, double timeUsed );

  // Returns a list of the tasks that have been completed.
  // The list is subsequently cleared and only updated with new tasks
  // that finish from the point of this call.
  void getFinishedTasks( vector<Task *> & finishedTasks );

  void assignThread( Task * task, const ProcessorGroup * pg );

  // Returns the percent of time that has been used by threads
  // for executing tasks.
  double getUtilization();

private:

  double            d_beginTime;
  int               d_numWorkers;
  int               d_numBusy;


  // List of the amount of time used by each thread for processing tasks.
  double          * d_timeUsed;

  // List of tasks that workers have finished
  vector<Task *>    d_finishedTasks;

  // All workers and the threadPool can only access the workerQueue serially
  Mutex           * d_workerQueueLock;

  // Used to unblock a worker thread after it has a task assigned to it.
  Mutex          ** d_workerReadyLocks; 

  // An array of Worker Threads.
  Worker         ** d_workers;

  // A stack of the ids of available (idle) threads
  stack<int>        d_availableThreads;
};
} // End namespace Uintah

