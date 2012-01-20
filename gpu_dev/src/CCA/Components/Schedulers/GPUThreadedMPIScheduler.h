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


#ifndef UINTAH_HOMEBREW_GPUTMPISCHEDULER_H
#define UINTAH_HOMEBREW_GPUTMPISCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>

namespace Uintah {


using std::vector;
using std::map;
using std::ofstream;

class Task;
class DetailedTask;
class TaskWorker;

/**************************************

CLASS
   GPUThreadedMPIScheduler
   
   Short description...

GENERAL INFORMATION

   GPUThreadedMPIScheduler.h

   Alan Humphrey, after Steven G. Parker & Qingyu Meng
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2011 SCI Group

KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
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
    
    void initiateGPUTask(DetailedTask* task, int iteration);

    void prepareTaskDeviceMemory(DetailedTask* dtask);

    void hostToDeviceVariableCopy (DetailedTask* dtask,
                                   const VarLabel* label,
                                   const Patch* patch,
                                   double* h_VarData);

    ConditionVariable      d_nextsignal;
    Mutex                  d_nextmutex;   //conditional wait mutex
    TaskWorker*            t_worker[16];  //workers
    Thread*                t_thread[16];
    Mutex                  dlbLock;       //load balancer lock
    

  private:
    
    Output*                oport_t;
    CommRecMPI             sends_[16+1];
    QueueAlg               taskQueueAlg_;
    int                    numThreads_;
    int                    numGPUs_;

    map<const VarLabel*, double*>    deviceVariableMemMap;

    GPUThreadedMPIScheduler(const GPUThreadedMPIScheduler&);
    GPUThreadedMPIScheduler& operator=(const GPUThreadedMPIScheduler&);
    void setNumGPUs(); // set by cudaGetDeviceCount
    int getAviableThreadNum();
  };

} // End namespace Uintah
   
#endif
