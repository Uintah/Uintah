/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UNIFIED_SCHEDULER_H
#define UNIFIED_SCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

#include <sci_defs/cuda_defs.h>

namespace Uintah {

class Task;
class DetailedTask;
class UnifiedSchedulerWorker;

/**************************************

CLASS
   UnifiedScheduler
   
   Multi-threaded/CPU/GPU/MPI scheduler

GENERAL INFORMATION

   UnifiedScheduler.h

   Qingyu Meng & Alan Humphrey
   Scientific Computing and Imaging Institute
   University of Utah

   
KEYWORDS
   Task Scheduler, Multi-threaded, CPU, GPU, MIC

DESCRIPTION
   This class is meant to be serve as a single-processor, unified, multi-threaded
   Uintah task scheduler to support, schedule and execute solely CPU tasks
   or some combination of CPU, GPU and MIC tasks when enabled.
  
WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks/components are GPU/MIC-enabled and/or thread-safe yet.
  
****************************************/
  class UnifiedScheduler : public MPIScheduler  {

  public:

    UnifiedScheduler( const ProcessorGroup * myworld, const Output * oport, UnifiedScheduler * parentScheduler = 0 );

    ~UnifiedScheduler();
    
    virtual void problemSetup(const ProblemSpecP& prob_spec, SimulationStateP& state);
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }
    
    virtual void initiateTask( DetailedTask * task, bool only_old_recvs, int abort_point, int iteration );

    virtual void runTask( DetailedTask* task, int iteration, int thread_id, Task::CallBackEvent event );

            void runTasks( int thread_id );
     
            void postMPISends( DetailedTask* task, int iteration, int thread_id );
    virtual void postMPIRecvs( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration );

    virtual void processMPIRecvs(int how_much);

    int  pendingMPIRecvs();

    ConditionVariable        d_nextsignal;           // conditional wait mutex
    Mutex                    d_nextmutex;            // mutex
    UnifiedSchedulerWorker*  t_worker[MAX_THREADS];  // the workers
    Thread*                  t_thread[MAX_THREADS];  // the threads themselves
    Mutex                    dlbLock;                // load balancer lock
    Mutex                    schedulerLock;          // scheduler lock (acquire and release quickly)
    Mutex                    waittimesLock;               // miscellaneous lock
    mutable CrowdMonitor     recvLock;               // multiple reader, single writer lock (pthread_rwlock_t wrapper)

#ifdef HAVE_CUDA

    cudaStream_t* getCudaStream(int device);

#endif

    /* thread shared data, needs lock protection when accessed */
    std::vector<int>           phaseTasks;
    std::vector<int>           phaseTasksDone;
    std::vector<DetailedTask*> phaseSyncTask;
    std::vector<int>           histogram;
    DetailedTasks*        dts;
    int   currentIteration;
    int   numTasksDone;
    int   ntasks;
    int   currphase;
    int   numPhase;
    bool  abort;
    int   abort_point;

  protected:

    virtual void verifyChecksum();

  private:
    
    int getAviableThreadNum();

    UnifiedScheduler(const UnifiedScheduler&);
    UnifiedScheduler& operator=(const UnifiedScheduler&);

    const Output * oport_t;
    CommRecMPI     sends_[MAX_THREADS];
    QueueAlg       taskQueueAlg_;
    int            numThreads_;

#ifdef HAVE_CUDA

    void gpuInitialize(bool reset=false);

    void postD2HCopies(DetailedTask* dtask);
    
    void preallocateDeviceMemory(DetailedTask* dtask);

    void postH2DCopies(DetailedTask* dtask);

    void createCudaStreams(int device, int numStreams = 1);

    void reclaimCudaStreams(DetailedTask* dtask);

    cudaError_t unregisterPageLockedHostMem();

    void freeCudaStreams();

    int           numDevices_;
    int           currentDevice_;

    std::vector<std::queue<cudaStream_t*> >  idleStreams;
    std::set<void*>  pinnedHostPtrs;

    // All are multiple reader, single writer locks (pthread_rwlock_t wrapper)
    mutable CrowdMonitor idleStreamsLock_;
    mutable CrowdMonitor d2hComputesLock_;
    mutable CrowdMonitor h2dRequiresLock_;

#endif
  };



class UnifiedSchedulerWorker : public Runnable {

public:
  
  UnifiedSchedulerWorker( UnifiedScheduler * scheduler, int thread_id );

  void assignTask( DetailedTask* task, int iteration);

  DetailedTask* getTask();

  void run();

  void quit(){d_quit=true;};

  double getWaittime();

  void resetWaittime(double start);
  
  friend class UnifiedScheduler;


private:

  int                    d_thread_id;
  UnifiedScheduler*      d_scheduler;
  bool                   d_idle;
  Mutex                  d_runmutex;
  ConditionVariable      d_runsignal;
  bool                   d_quit;
  double                 d_waittime;
  double                 d_waitstart;
  int                    d_rank;
  CommRecMPI             d_sends_;
};

} // End namespace Uintah
   
#endif
