/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_HOMEBREW_TMPISCHEDULER2_H
#define UINTAH_HOMEBREW_TMPISCHEDULER2_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

namespace Uintah {


using std::vector;
using std::ofstream;
using std::set;

class Task;
class DetailedTask;
class SchedulerWorker;

/**************************************

CLASS
   ThreadedMPIScheduler2
   
   Short description...

GENERAL INFORMATION

   ThreadedMPIScheduler2.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ThreadedMPIScheduler2 : public MPIScheduler  {
  public:
    ThreadedMPIScheduler2(const ProcessorGroup* myworld, Output* oport, ThreadedMPIScheduler2* parentScheduler = 0);
     ~ThreadedMPIScheduler2();
    
    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              SimulationStateP& state);
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep();}
    
    
    void initiateTask( DetailedTask          * task,
			       bool only_old_recvs, int abort_point, int iteration);

    void runTask( DetailedTask* task, int iteration, int t_id);
    
    void postMPISends( DetailedTask* task, int iteration, int t_id);
    
    void postMPIRecvs( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration);

    enum { TEST, WAIT_ONCE, WAIT_ALL};

    void processMPIRecvs(int how_much);    
    int  pendingMPIRecvs();    
    //void assignTask( DetailedTask* task, int iteration);

    void runTasks(int t_id);
    
    ConditionVariable     d_nextsignal;
    Mutex                  d_nextmutex;   //conditional wait mutex
    SchedulerWorker*            t_worker[MAX_THREADS];  //workers
    Thread*                t_thread[MAX_THREADS]; 
    Mutex                  dlbLock;   //load balancer lock
    Mutex                  schedulerLock; //scheduler lock
    mutable CrowdMonitor   recvLock;
    
    /* thread shared data, need lock protection when accessing them */
    DetailedTasks* dts; 
    int curriteration;
    int numTasksDone;
    int ntasks;
    int currphase;
    int numPhase;
    vector<int> phaseTasks;
    vector<int> phaseTasksDone;
    vector<DetailedTask *> phaseSyncTask;
    bool abort;
    int abort_point;
    set< DetailedTask * > pending_tasks;
    vector<int> histogram;

    
  private:
    
    Output*       oport_t;
    CommRecMPI            sends_[MAX_THREADS];
    ThreadedMPIScheduler2(const ThreadedMPIScheduler2&);
    ThreadedMPIScheduler2& operator=(const ThreadedMPIScheduler2&);
    
    QueueAlg taskQueueAlg_;
    int numThreads_;
    int getAviableThreadNum();
  };

class SchedulerWorker : public Runnable { 

public:
  
  SchedulerWorker(ThreadedMPIScheduler2* scheduler, int id);

  void assignTask( DetailedTask* task, int iteration);

  DetailedTask* getTask();

  void run();

  void quit(){d_quit=true;};

  double getWaittime();
  void resetWaittime(double start);
  
  friend class ThreadedMPIScheduler2;


private:
  int                    d_id;
  ThreadedMPIScheduler2*  d_scheduler;
  bool                   d_idle;
  Mutex d_runmutex;
  ConditionVariable d_runsignal;
  bool                   d_quit;
  double                 d_waittime;
  double                 d_waitstart;
  int                    d_rank;
  CommRecMPI            d_sends_;
};

} // End namespace Uintah
   
#endif
