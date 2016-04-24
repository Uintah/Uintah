/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CCA_COMPONENTS_SCHEDULERS_THREADEDMPISCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_THREADEDMPISCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace Uintah {

class Task;
class DetailedTask;
class TaskWorker;

/**************************************

 CLASS
   ThreadedMPIScheduler


 GENERAL INFORMATION
   ThreadedMPIScheduler.h

   Qingyu Meng & Alan Humphrey
   Scientific Computing and Imaging Institute
   University of Utah


 KEYWORDS
   Task Scheduler, Multi-threaded MPI


 DESCRIPTION
   A multi-threaded MPI scheduler that uses a combination of MPI + Pthreads, with dynamic
   scheduling with non-deterministic, out-of-order execution of tasks at runtime. One
   MPI rank per multi-core node. Pthreads are pinned to individual CPU cores where
   tasks are executed. Uses a centralized model using 1 control thread and "nthreads − 1"
   task execution threads. The control thread assigns tasks and processes MPI receives.
   Threads have shared access to the DataWarehouse.

   This Scheduler using a shared memory model on-node, that is, 1 MPI process per node and "-nthreads"
   number of Pthreads to execute tasks on available CPU cores.


 WARNING
   Tasks must be thread-safe when using this Scheduler.
   Requires MPI_THREAD_MULTIPLE support.

 ****************************************/

class ThreadedMPIScheduler : public MPIScheduler {

  public:

    ThreadedMPIScheduler( const ProcessorGroup* myworld, const Output* oport, ThreadedMPIScheduler* parentScheduler = 0 );

    virtual ~ThreadedMPIScheduler();

    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state );

    virtual SchedulerP createSubScheduler();

    virtual void execute( int tgnum = 0, int iteration = 0 ) ;

    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }

    friend class TaskWorker;

  private:

    // Disable copy and assignment
    ThreadedMPIScheduler( const ThreadedMPIScheduler& );
    ThreadedMPIScheduler& operator=( const ThreadedMPIScheduler& );

    void assignTask( DetailedTask* task, int iteration );

    int getAvailableThreadNum();

    std::condition_variable  d_nextsignal;
    std::mutex               d_nextmutex;             // conditional wait mutex
    TaskWorker*              t_worker[MAX_THREADS];   // workers
    std::thread*             t_thread[MAX_THREADS];   // actual threads
    QueueAlg                 taskQueueAlg_;
    int                      numThreads_;
};


class TaskWorker {

  public:

    TaskWorker( ThreadedMPIScheduler* scheduler, int thread_id );

    ~TaskWorker();

    void run();

    void quit() { d_quit = true; };

    double getWaittime();

    void resetWaittime( double start );

    friend class ThreadedMPIScheduler;

  private:

    ThreadedMPIScheduler*    d_scheduler;
    DetailedTask*            d_task;
    std::condition_variable  d_runsignal;
    std::mutex               d_runmutex;
    bool                     d_quit;
    int                      d_thread_id;
    int                      d_rank;
    int                      d_iteration;
    double                   d_waittime;
    double                   d_waitstart;
    size_t                   d_numtasks;
};

}  // End namespace Uintah

#endif // End CCA_COMPONENTS_SCHEDULERS_THREADEDMPISCHEDULER_H
