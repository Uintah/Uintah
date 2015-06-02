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

#ifndef CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

#include <sci_defs/cuda_defs.h>

#include <string>

namespace Uintah {

class Task;
class DetailedTask;
class UnifiedSchedulerWorker;

/**************************************

CLASS
   UnifiedScheduler
   

GENERAL INFORMATION
   UnifiedScheduler.h

   Qingyu Meng & Alan Humphrey
   Scientific Computing and Imaging Institute
   University of Utah

   
KEYWORDS
   Task Scheduler, Multi-threaded MPI, CPU, GPU, MIC

DESCRIPTION
   A multi-threaded scheduler that uses a combination of MPI + Pthreads
   and offers support for GPU tasks. Dynamic scheduling with non-deterministic,
   out-of-order execution of tasks at runtime. One MPI rank per multi-core node.
   Pthreads are pinned to individual CPU cores where these tasks are executed.
   Uses a decentralized model wherein all threads can access task queues,
   processes there own MPI send and recvs, with shared access to the DataWarehouse.

   Uintah task scheduler to support, schedule and execute solely CPU tasks
   or some combination of CPU, GPU and MIC tasks when enabled.
  
WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks/components are GPU/MIC-enabled and/or thread-safe yet.
   
   Requires MPI_THREAD_MULTIPLE support.
  
****************************************/

class UnifiedScheduler : public MPIScheduler  {

  public:

    UnifiedScheduler( const ProcessorGroup* myworld, const Output* oport, UnifiedScheduler* parentScheduler = 0 );

    virtual ~UnifiedScheduler();
    
    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state );
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute( int tgnum = 0, int iteration = 0 );
    
    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }
    
    virtual void runTask( DetailedTask* task, int iteration, int thread_id, Task::CallBackEvent event );

    void runTasks( int thread_id );

    friend class UnifiedSchedulerWorker;

    static const int bufferPadding = 32;

  private:

    // Disable copy and assignment
    UnifiedScheduler( const UnifiedScheduler& );
    UnifiedScheduler& operator=( const UnifiedScheduler& );

    int getAvailableThreadNum();

    int  pendingMPIRecvs();

    ConditionVariable          d_nextsignal;           // conditional wait mutex
    Mutex                      d_nextmutex;            // next mutex
    Mutex                      schedulerLock;          // scheduler lock (acquire and release quickly)
    UnifiedSchedulerWorker*    t_worker[MAX_THREADS];  // the workers
    Thread*                    t_thread[MAX_THREADS];  // the threads themselves

    // thread shared data, needs lock protection when accessed
    std::vector<int>           phaseTasks;
    std::vector<int>           phaseTasksDone;
    std::vector<DetailedTask*> phaseSyncTask;
    std::vector<int>           histogram;
    DetailedTasks*             dts;

    QueueAlg taskQueueAlg_;
    int      currentIteration;
    int      numTasksDone;
    int      ntasks;
    int      currphase;
    int      numPhases;
    bool     abort;
    int      abort_point;
    int      numThreads_;

#ifdef HAVE_CUDA


    void prepareGPUDependencies(DetailedTask* task, DependencyBatch* batch, const VarLabel* pos_var,
           OnDemandDataWarehouse* dw, OnDemandDataWarehouse* old_dw, const DetailedDep* dep, LoadBalancer* lb);

    void gpuInitialize( bool reset=false );

    void performInternalGhostCellCopies(DetailedTask* dtask);

    void initiateH2DCopies(DetailedTask* dtask);

    void markDeviceRequiresDataAsValid(DetailedTask* dtask);

    void markDeviceComputesDataAsValid(DetailedTask* dtask);

    void markHostRequiresDataAsValid(DetailedTask* dtask);

    bool initiateD2H(DetailedTask* dtask);

    void copyAllDataD2H(DetailedTask* dtask);

    void processD2HCopies(DetailedTask* dtask);

    void postD2HCopies( DetailedTask* dtask );
    
    void postH2DCopies(DetailedTask* dtask);

    void preallocateDeviceMemory( DetailedTask* dtask );

    void createCudaStreams(int numStreams, int device);

    void reclaimCudaStreams( DetailedTask* dtask );

    void addCudaStream(cudaStream_t* stream, int device);

    void addCudaEvent(cudaEvent_t* event, int device);

    void freeCudaStreams();

    cudaError_t freeDeviceRequiresMem();

    cudaStream_t* getCudaStream(int device);

    cudaError_t freeComputesMem();

    cudaError_t unregisterPageLockedHostMem();

    void freeCudaEvents();

    void clearGpuDBMaps();

    void assignDevice(DetailedTask* task);

    void prepareGPUDependencies( DetailedTask* task, int iteration, int t_id);

    void copyGPUInternalDependencies(DetailedTask * task, int iteration, int t_id);

    struct GPUGridVariableInfo {
      DetailedTask* dtask;
      double*       ptr;
      IntVector     size;
      int           device;
      GPUGridVariableInfo(DetailedTask* _dtask, double* _ptr, IntVector _size, int _device)
        : dtask(_dtask), ptr(_ptr), size(_size), device(_device) {
      }
    };

    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo> deviceRequiresPtrs;
    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo> deviceComputesPtrs;
    std::map<std::string, GPUGridVariableInfo> deviceComputesTemporaryPtrs;
    std::vector<VarLabel*> temporaryVarLabels;

    std::vector<GPUGridVariableInfo> deviceRequiresAllocationPtrs;
    std::vector<GPUGridVariableInfo> deviceComputesAllocationPtrs;
    std::vector<double*> hostComputesAllocationPtrs;

    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo> hostRequiresPtrs;
    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo> hostComputesPtrs;
    std::vector<std::queue<cudaEvent_t*> >   idleEvents;
    int  numDevices_;
    int  currentDevice_;

    /* thread shared data, needs lock protection when accessed */
    std::vector<std::queue<cudaStream_t*> >  idleStreams;
    std::set<void*>                          pinnedHostPtrs;
    std::vector< std::string >               materialsNames;

    // All are multiple reader, single writer locks (pthread_rwlock_t wrapper)
    mutable CrowdMonitor d2hComputesLock_;
    mutable CrowdMonitor idleStreamsLock_;
    mutable CrowdMonitor h2dRequiresLock_;
    /*mutable CrowdMonitor deviceComputesLock_;
    mutable CrowdMonitor hostComputesLock_;
    mutable CrowdMonitor deviceRequiresLock_;
    mutable CrowdMonitor hostRequiresLock_;
    mutable CrowdMonitor deviceComputesAllocationLock_;
    mutable CrowdMonitor hostComputesAllocationLock_;
    mutable CrowdMonitor deviceComputesTemporaryLock_;*/

    struct labelPatchMatlDependency {
      std::string     label;
      int        patchID;
      int        matlIndex;
      Task::DepType    depType;

      labelPatchMatlDependency(const char * label, int patchID, int matlIndex, Task::DepType depType) {
        this->label = label;
        this->patchID = patchID;
        this->matlIndex = matlIndex;
        this->depType = depType;
      }
      //This so it can be used in an STL map
      bool operator<(const labelPatchMatlDependency& right) const {
        if (this->label < right.label) {
          return true;
        } else if (this->label == right.label && (this->patchID < right.patchID)) {
          return true;
        } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndex < right.matlIndex)) {
          return true;
        } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndex == right.matlIndex) && (this->depType < right.depType)) {
          return true;
        } else {
          return false;
        }

      }

    };

#endif
};



class UnifiedSchedulerWorker : public Runnable {

public:
  
  UnifiedSchedulerWorker( UnifiedScheduler* scheduler, int thread_id );

  void run();

  void quit() { d_quit = true; };

  double getWaittime();

  void resetWaittime( double start );
  
  friend class UnifiedScheduler;

private:

  UnifiedScheduler*      d_scheduler;
  ConditionVariable      d_runsignal;
  Mutex                  d_runmutex;
  bool                   d_quit;
  bool                   d_idle;
  int                    d_thread_id;
  int                    d_rank;
  double                 d_waittime;
  double                 d_waitstart;


};

} // End namespace Uintah
   
#endif // End CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H
