/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

namespace Uintah {

class Task;
class DetailedTask;
class TaskWorker;

/**************************************

 CLASS
   ThreadedMPIScheduler


 GENERAL INFORMATION

   ThreadedMPIScheduler.h


 KEYWORDS
  Multi-threaded MPI Scheduler


 DESCRIPTION
  This Scheduler using a shared memory model on-node, that is, 1 MPI process per node and "-nthreads"
  number of Pthreads to execute tasks on available CPU cores.


 WARNING
   Tasks must be thread-safe when using this Scheduler.

 ****************************************/
class ThreadedMPIScheduler : public MPIScheduler {

  public:

    ThreadedMPIScheduler(const ProcessorGroup* myworld,
                         const Output* oport,
                         ThreadedMPIScheduler* parentScheduler = 0);

    ~ThreadedMPIScheduler();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              SimulationStateP& state);

    virtual SchedulerP createSubScheduler();

    virtual void execute(int tgnum = 0,
                         int iteration = 0);

    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }

    void runTask(DetailedTask* task,
                 int iteration,
                 int t_id = 0);

    void postMPISends(DetailedTask* task,
                      int iteration,
                      int t_id);

    void assignTask(DetailedTask* task,
                    int iteration);

    ConditionVariable d_nextsignal;
    Mutex             d_nextmutex;             //conditional wait mutex
    TaskWorker*       t_worker[MAX_THREADS];  //workers
    Thread*           t_thread[MAX_THREADS];
    Mutex             dlbLock;                //load balancer lock

  private:

    Output*    oport_t;
    CommRecMPI sends_[MAX_THREADS];
    
    // Disable copy and assignment
    ThreadedMPIScheduler(const ThreadedMPIScheduler&);
    ThreadedMPIScheduler& operator=(const ThreadedMPIScheduler&);

    QueueAlg taskQueueAlg_;
    int      numThreads_;
    int getAviableThreadNum();
};

class TaskWorker : public Runnable {

  public:

    TaskWorker(ThreadedMPIScheduler* scheduler,
               int id);

    void assignTask(DetailedTask* task,
                    int iteration);

    DetailedTask* getTask();

    void run();

    void quit() { d_quit = true; };

    double getWaittime();

    void resetWaittime(double start);

    friend class ThreadedMPIScheduler;

  private:

    int d_id;
    ThreadedMPIScheduler* d_scheduler;
    DetailedTask*         d_task;
    int                   d_iteration;
    Mutex                 d_runmutex;
    ConditionVariable     d_runsignal;
    bool                  d_quit;
    CommRecMPI            d_sends_;
    double                d_waittime;
    double                d_waitstart;
    int                   d_rank;
};

}  // End namespace Uintah

#endif // End CCA_COMPONENTS_SCHEDULERS_THREADEDMPISCHEDULER_H
