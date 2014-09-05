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

#ifndef UINTAH_HOMEBREW_GPUTHREADEDMPISCHEDULER_H
#define UINTAH_HOMEBREW_GPUTHREADEDMPISCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

namespace Uintah {


using std::vector;
using std::map;
using std::queue;
using std::set;
using std::ofstream;

class Task;
class DetailedTask;
class GPUTaskWorker;

/**************************************

CLASS
   GPUThreadedMPIScheduler
   
   GPU/Multi-thread/MPI scheduler

GENERAL INFORMATION

   GPUThreadedMPIScheduler.h

   Alan Humphrey, after Qingyu Meng & Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Task Scheduler, Multi-threaded, GPU

DESCRIPTION
   This class extends the existing multi-threaded MPI scheduler to
   support, schedule and execute tasks that have been GPU-enabled.
  
WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks are GPU-enabled and thread-safe yet.
  
****************************************/
  class GPUThreadedMPIScheduler : public MPIScheduler  {

  public:

    GPUThreadedMPIScheduler(const ProcessorGroup* myworld, Output* oport, GPUThreadedMPIScheduler* parentScheduler = 0);
    
    ~GPUThreadedMPIScheduler();

    virtual void problemSetup(const ProblemSpecP& prob_spec, SimulationStateP& state);
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }
    
    void runTask(DetailedTask* task, int iteration, int t_id = 0);
    
    void runGPUTask(DetailedTask* task, int iteration, int t_id = 0);

    void postMPISends(DetailedTask* task, int iteration, int t_id);
    
    void assignTask(DetailedTask* task, int iteration);
    
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

    enum CopyType {H2D, D2H};

    ConditionVariable     d_nextsignal;
    Mutex                 d_nextmutex;            //conditional wait mutex
    GPUTaskWorker*        t_worker[MAX_THREADS];  //workers
    Thread*               t_thread[MAX_THREADS];
    Mutex                 dlbLock;                //load balancer lock
    

  private:
    
    GPUThreadedMPIScheduler(const GPUThreadedMPIScheduler&);

    GPUThreadedMPIScheduler& operator=(const GPUThreadedMPIScheduler&);

    void gpuInitialize();

    int getAviableThreadNum();

    void initiateH2DRequiresCopies(DetailedTask* dtask, int iteration);

    void initiateH2DComputesCopies(DetailedTask* dtask, int iteration);

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

    Output*                oport_t;
    CommRecMPI             sends_[MAX_THREADS];
    QueueAlg               taskQueueAlg_;
    int                    numThreads_;
    int                    numGPUs_;
    int                    currentGPU_;

    struct GPUGridVariable {
      DetailedTask* dtask;
      double*       ptr;
      IntVector     size;
      int           device;
      GPUGridVariable(DetailedTask* _dtask, double* _ptr, IntVector _size, int _device)
        : dtask(_dtask), ptr(_ptr), size(_size), device(_device) {
      }
    };

    map<VarLabelMatl<Patch>, GPUGridVariable> deviceRequiresPtrs; // simply uses cudaFree on these device allocations

    map<VarLabelMatl<Patch>, GPUGridVariable> deviceComputesPtrs; // simply uses cudaFree on these device allocations

    map<VarLabelMatl<Patch>, GPUGridVariable> hostRequiresPtrs;   // unregister all requires host pointers that were page-locked

    map<VarLabelMatl<Patch>, GPUGridVariable> hostComputesPtrs;   // unregister all computes host pointers that were page-locked

    set<double*> pinnedHostPtrs;

    vector<queue<cudaStream_t*> >  idleStreams;

    vector<queue<cudaEvent_t*> >   idleEvents;

  };

  class GPUTaskWorker : public Runnable {

    public:

  	GPUTaskWorker(GPUThreadedMPIScheduler* scheduler, int id);

      ~GPUTaskWorker();

      void assignTask(DetailedTask* task, int iteration);

      DetailedTask* getTask();

      void run();

      void quit(){d_quit=true;};

      double getWaittime();

      void resetWaittime(double start);

      friend class GPUThreadedMPIScheduler;


    private:
      int                       d_id;
      GPUThreadedMPIScheduler*  d_schedulergpu;
      DetailedTask*             d_task;
      int                       d_iteration;
      Mutex                     d_runmutex;
      ConditionVariable         d_runsignal;
      bool                      d_quit;
      CommRecMPI                d_sends_;
      double                    d_waittime;
      double                    d_waitstart;
      int                       d_rank;
    };

} // End namespace Uintah
   
#endif
