/*

The MIT License

Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
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


#ifndef UNIFIED_SCHEDULER_H
#define UNIFIED_SCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

#include <sci_defs/cuda_defs.h>

namespace Uintah {


using std::vector;
using std::map;
using std::queue;
using std::set;
using std::ofstream;

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

   Copyright (C) 2012 SCI Group

KEYWORDS
   Task Scheduler, Multi-threaded, CPU, GPU

DESCRIPTION
   This class is meant to be serve as a single, unified, multi-threaded
   Uintah task scheduler to support, schedule and execute solely CPU tasks
   or both CPU and GPU tasks when enabled.
  
WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks/components are GPU-enabled and/or thread-safe yet.
  
****************************************/
  class UnifiedScheduler : public MPIScheduler  {

  public:

    UnifiedScheduler(const ProcessorGroup* myworld, Output* oport, UnifiedScheduler* parentScheduler = 0);

    ~UnifiedScheduler();
    
    virtual void problemSetup(const ProblemSpecP& prob_spec, SimulationStateP& state);
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }
    
    void initiateTask( DetailedTask * task, bool only_old_recvs, int abort_point, int iteration);

    void runTask( DetailedTask* task, int iteration, int t_id);

    void runTasks(int t_id);
    
    void postMPISends( DetailedTask* task, int iteration, int t_id);
    
    void postMPIRecvs( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration);

    void processMPIRecvs(int how_much);

    int  pendingMPIRecvs();

    enum { TEST, WAIT_ONCE, WAIT_ALL };

    ConditionVariable        d_nextsignal;
    Mutex                    d_nextmutex;            // conditional wait mutex
    UnifiedSchedulerWorker*  t_worker[MAX_THREADS];  // workers
    Thread*                  t_thread[MAX_THREADS];
    Mutex                    dlbLock;                // load balancer lock
    Mutex                    schedulerLock;          // scheduler lock
    mutable CrowdMonitor     recvLock;

#ifdef HAVE_CUDA

    void runTasksGPU(int t_id);

    double* getDeviceRequiresPtr(const VarLabel* label, int matlIndex, const Patch* patch);

    double* getDeviceComputesPtr(const VarLabel* label, int matlIndex, const Patch* patch);

    double* getHostRequiresPtr(const VarLabel* label, int matlIndex, const Patch* patch);

    double* getHostComputesPtr(const VarLabel* label, int matlIndex, const Patch* patch);

    IntVector getDeviceRequiresSize(const VarLabel* label, int matlIndex, const Patch* patch);

    IntVector getDeviceComputesSize(const VarLabel* label, int matlIndex, const Patch* patch);

    void requestD2HCopy(const VarLabel* label, int matlIndex, const Patch* patch, cudaStream_t* stream, cudaEvent_t* event);

    void createCudaStreams(int numStreams, int device);

    void createCudaEvents(int numEvents, int device);

    cudaStream_t* getCudaStream(int device);

    cudaEvent_t* getCudaEvent(int device);

    void addCudaStream(cudaStream_t* stream, int device);

    void addCudaEvent(cudaEvent_t* event, int device);

    enum CopyType { H2D, D2H };

#endif

    /* thread shared data, needs lock protection when accessed */
    vector<int>           phaseTasks;
    vector<int>           phaseTasksDone;
    vector<DetailedTask*> phaseSyncTask;
    vector<int>           histogram;
    DetailedTasks*        dts;
    int   currentIteration;
    int   numTasksDone;
    int   ntasks;
    int   currphase;
    int   numPhase;
    bool  abort;
    int   abort_point;

  private:
    
    int getAviableThreadNum();

    UnifiedScheduler(const UnifiedScheduler&);

    UnifiedScheduler& operator=(const UnifiedScheduler&);
    Output*       oport_t;
    CommRecMPI    sends_[MAX_THREADS];
    QueueAlg      taskQueueAlg_;
    int           numThreads_;

#ifdef HAVE_CUDA

    void gpuInitialize();

    void initiateH2DRequiresCopies(DetailedTask* dtask);

    void initiateH2DComputesCopies(DetailedTask* dtask);

    void h2dRequiresCopy (DetailedTask* dtask, const VarLabel* label, int matlIndex, const Patch* patch, IntVector size, double* h_reqData);

    void h2dComputesCopy (DetailedTask* dtask, const VarLabel* label, int matlIndex, const Patch* patch, IntVector size, double* h_compData);

    void reclaimStreams(DetailedTask* dtask, CopyType type);

    void reclaimEvents(DetailedTask* dtask, CopyType type);

    cudaError_t freeDeviceRequiresMem();

    cudaError_t freeDeviceComputesMem();

    cudaError_t unregisterPageLockedHostMem();

    void clearCudaStreams();

    void clearCudaEvents();

    void clearMaps();

    struct GPUGridVariable {
      DetailedTask* dtask;
      double*       ptr;
      IntVector     size;
      int           device;
      GPUGridVariable(DetailedTask* _dtask, double* _ptr, IntVector _size, int _device)
        : dtask(_dtask), ptr(_ptr), size(_size), device(_device) {
      }
    };

    map<VarLabelMatl<Patch>, GPUGridVariable> deviceRequiresPtrs;
    map<VarLabelMatl<Patch>, GPUGridVariable> deviceComputesPtrs;
    map<VarLabelMatl<Patch>, GPUGridVariable> hostRequiresPtrs;
    map<VarLabelMatl<Patch>, GPUGridVariable> hostComputesPtrs;
    vector<queue<cudaStream_t*> >  idleStreams;
    vector<queue<cudaEvent_t*> >   idleEvents;
    set<double*>  pinnedHostPtrs;
    int           numGPUs_;
    int           currentGPU_;

    mutable CrowdMonitor deviceComputesLock_;
    mutable CrowdMonitor hostComputesLock_;
    mutable CrowdMonitor deviceRequiresLock_;
    mutable CrowdMonitor hostRequiresLock_;
    mutable CrowdMonitor idleStreamsLock_;
    mutable CrowdMonitor idleEventsLock_;
    mutable CrowdMonitor h2dComputesLock_;
    mutable CrowdMonitor h2dRequiresLock_;

#endif
  };



class UnifiedSchedulerWorker : public Runnable {

public:
  
  UnifiedSchedulerWorker(UnifiedScheduler* scheduler, int id);

  void assignTask( DetailedTask* task, int iteration);

  DetailedTask* getTask();

  void run();

  void quit(){d_quit=true;};

  double getWaittime();

  void resetWaittime(double start);
  
  friend class UnifiedScheduler;


private:

  int                    d_id;
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
