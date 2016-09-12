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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

#include <CCA/Components/Schedulers/DetailedDependency.h>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Grid/Variables/ScrubItem.h>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
  #include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <atomic>
#include <list>
#include <queue>
#include <vector>
#include <map>
#include <set>

#include <sci_defs/cuda_defs.h>

namespace Uintah {

class ProcessorGroup;
class DependencyBatch;
class DataWarehouse;
class DetailedTask;
class DetailedTasks;
class TaskGraph;
class SchedulerCommon;


using ParticleExchangeVar = std::map<int, std::set<PSPatchMatlGhostRange> >;
using ScrubCountTable     = FastHashTable<ScrubItem>;
using DepCommCond         = DetailedDep::CommCondition;


enum ProfileType {
    Normal
  , Fine
};


enum QueueAlg {
    FCFS
  , Stack
  , Random
  , MostChildren
  , LeastChildren
  , MostAllChildren
  , LeastAllChildren
  , MostMessages
  , LeastMessages
  , MostL2Children
  , LeastL2Children
  , CritialPath
  , PatchOrder
  , PatchOrderRandom
};


struct InternalDependency {
  InternalDependency(       DetailedTask * prerequisiteTask
                    ,       DetailedTask * dependentTask
                    , const VarLabel     * var
                    ,       long           satisfiedGeneration
                    )
    : prerequisiteTask(prerequisiteTask)
    , dependentTask(dependentTask)
    , satisfiedGeneration(satisfiedGeneration)
  {
    addVarLabel(var);
  }

  void addVarLabel(const VarLabel* var)
  {
    vars.insert(var);
  }

  DetailedTask                                 * prerequisiteTask;
  DetailedTask                                 * dependentTask;
  std::set<const VarLabel*, VarLabel::Compare>   vars;
  unsigned long                                  satisfiedGeneration;
};

#ifdef HAVE_CUDA
  struct TaskGpuDataWarehouses {
  public:
    GPUDataWarehouse* TaskGpuDW[2];
};
#endif

class DetailedTask {

public:

  DetailedTask(       Task           * task
              , const PatchSubset    * patches
              , const MaterialSubset * matls
              ,       DetailedTasks  * taskGroup
              );

  ~DetailedTask();

  void setProfileType( ProfileType type )
  {
    d_profileType = type;
  }

  ProfileType getProfileType()
  {
    return d_profileType;
  }

  void doit( const ProcessorGroup                      * pg
           ,       std::vector<OnDemandDataWarehouseP> & oddws
           ,       std::vector<DataWarehouseP>         & dws
           ,       Task::CallBackEvent                   event = Task::CPU
           );

  // Called after doit and MPI data sent (packed in buffers) finishes.
  // Handles internal dependencies and scrubbing. Called after doit finishes.
  void done( std::vector<OnDemandDataWarehouseP> & dws );

  std::string getName() const;

  const Task* getTask() const
  {
    return task;
  }

  const PatchSubset* getPatches() const
  {
    return patches;
  }

  const MaterialSubset* getMaterials() const
  {
    return matls;
  }

  void assignResource(int idx)
  {
    resourceIndex = idx;
  }

  int getAssignedResourceIndex() const
  {
    return resourceIndex;
  }

  void assignStaticOrder(int i)
  {
    staticOrder = i;
  }

  int getStaticOrder() const
  {
    return staticOrder;
  }

  DetailedTasks* getTaskGroup() const
  {
    return taskGroup;
  }

  std::map<DependencyBatch*, DependencyBatch*>& getRequires()
  {
    return reqs;
  }

  std::map<DependencyBatch*, DependencyBatch*>& getInternalRequires()
  {
    return internal_reqs;
  }

  DependencyBatch* getComputes() const
  {
    return comp_head;
  }

  DependencyBatch* getInternalComputes() const
  {
    return internal_comp_head;
  }

  void findRequiringTasks( const VarLabel * var , std::list<DetailedTask*> & requiringTasks );

  void emitEdges( ProblemSpecP edgesElement );

  bool addInternalRequires( DependencyBatch * req );

  void addInternalComputes( DependencyBatch * comp );

  bool addRequires( DependencyBatch * req );

  void addComputes( DependencyBatch * comp );

  void addInternalDependency( DetailedTask * prerequisiteTask, const VarLabel * var );

  // external dependencies will count how many messages this task is waiting for.
  // When it hits 0, we can add it to the  DetailedTasks::mpiCompletedTasks list.
  void resetDependencyCounts();

  void markInitiated()
  {
    initiated_.store(true, std::memory_order_seq_cst);
  }

  void incrementExternalDepCount()
  {
    externalDependencyCount_.fetch_add(1, std::memory_order_seq_cst);
  }

  void decrementExternalDepCount()
  {
    externalDependencyCount_.fetch_sub(1, std::memory_order_seq_cst);
  }

  void checkExternalDepCount();

  int getExternalDepCount()
  {
    return externalDependencyCount_.load(std::memory_order_seq_cst);
  }

  bool areInternalDependenciesSatisfied()
  {
    return (numPendingInternalDependencies == 0);
  }

#ifdef HAVE_CUDA

  void assignDevice( unsigned int device );

  // Most tasks will only run on one device.
  // But some, such as the data archiver task or send old data could run on multiple devices.
  // This is not a good idea.  A task should only run on one device.  But the capability for a task
  // to run on multiple nodes exists.
  std::set<unsigned int> getDeviceNums() const;

  std::map<unsigned int, TaskGpuDataWarehouses> TaskGpuDWs;

  void setCudaStreamForThisTask( unsigned int deviceNum, cudaStream_t * s );

  void clearCudaStreamsForThisTask();

  bool checkCudaStreamDoneForThisTask( unsigned int deviceNum ) const;

  bool checkAllCudaStreamsDoneForThisTask() const;

  void setTaskGpuDataWarehouse( unsigned int       deviceNum
                              , Task::WhichDW      DW
                              , GPUDataWarehouse * TaskDW
                              );

  GPUDataWarehouse* getTaskGpuDataWarehouse( unsigned int deviceNum, Task::WhichDW DW );

  void deleteTaskGpuDataWarehouses();

  cudaStream_t* getCudaStreamForThisTask( unsigned int deviceNum ) const;

  DeviceGridVariables& getDeviceVars()
  {
    return deviceVars;
  }

  DeviceGridVariables& getTaskVars()
  {
    return taskVars;
  }

  DeviceGhostCells& getGhostVars()
  {
    return ghostVars;
  }

  DeviceGridVariables& getVarsToBeGhostReady()
  {
    return varsToBeGhostReady;
  }

  DeviceGridVariables& getVarsBeingCopiedByTask()
  {
    return varsBeingCopiedByTask;
  }

  void clearPreparationCollections();

  void addTempHostMemoryToBeFreedOnCompletion( void * ptr );

  void addTempCudaMemoryToBeFreedOnCompletion( unsigned int device_ptr, void * ptr );

  void deleteTemporaryTaskVars();

#endif


protected:

  friend class TaskGraph;


private:

  // eliminate copy, assignment and move
  DetailedTask(const DetailedTask &)            = delete;
  DetailedTask& operator=(const DetailedTask &) = delete;
  DetailedTask(DetailedTask &&)                 = delete;
  DetailedTask& operator=(DetailedTask &&)      = delete;

  // called by done()
  void scrub(std::vector<OnDemandDataWarehouseP> &);

  Task                                         * task;
  const PatchSubset                            * patches;
  const MaterialSubset                         * matls;
  std::map<DependencyBatch*, DependencyBatch*>   reqs;
  std::map<DependencyBatch*, DependencyBatch*>   internal_reqs;
  DependencyBatch                              * comp_head { nullptr };
  DependencyBatch                              * internal_comp_head { nullptr };
  DetailedTasks                                * taskGroup;

  std::atomic<bool> initiated_ { false };
  std::atomic<bool> externallyReady_ { false };
  std::atomic<int>  externalDependencyCount_ { 0 };

  mutable std::string name_;  // doesn't get set until getName() is called the first time.

  // Called when prerequisite tasks (dependencies) call done.
  void dependencySatisfied(InternalDependency * dep);

  // Internal dependencies are dependencies within the same process.
  std::list<InternalDependency> internalDependencies;

  // internalDependents will point to InternalDependency's in the
  // internalDependencies list of the requiring DetailedTasks.
  std::map<DetailedTask*, InternalDependency*> internalDependents;

  unsigned long numPendingInternalDependencies { 0 };

  int resourceIndex { -1 };
  int staticOrder   { -1 };

  bool operator<(const DetailedTask & other);

  // specifies the type of task this is:
  //   * Normal executes on either the patches cells or the patches coarse cells
  //   * Fine   executes on the patches fine cells (for example coarsening)
  ProfileType d_profileType { Normal };

#ifdef HAVE_CUDA

  bool         deviceExternallyReady_{false};
  bool         completed_{false};
  unsigned int deviceNum_{0};

  std::set<unsigned int>                deviceNums_;
  std::map<unsigned int, cudaStream_t*> d_cudaStreams;

  // Store information about each set of grid variables.
  // This will help later when we figure out the best way to store data into the GPU.
  // It may be stored contiguously.  It may handle material data.  It just helps to gather it all up
  // into a collection prior to copying data.
  DeviceGridVariables deviceVars;  // Holds variables that will need to be copied into the GPU
  DeviceGridVariables taskVars;    // Holds variables that will be needed for a GPU task (a Task DW has a snapshot of
                                   // all important pointer info from the host-side GPU DW)
  DeviceGhostCells ghostVars;      // Holds ghost cell meta data copy information

  DeviceGridVariables varsToBeGhostReady;  // Holds a list of vars this task is managing to ensure their ghost cells will be ready.
                                           // This means this task is the exlusive ghost cell gatherer and ghost cell validator for any
                                           // label/patch/matl/level vars it has listed in here
                                           // But it is NOT the exclusive copier.  Because some ghost cells from one patch may be used by
                                           // two or more destination patches.  We only want to copy ghost cells once.

  DeviceGridVariables varsBeingCopiedByTask;  // Holds a list of the vars that this task is actually copying into the GPU.

  struct gpuMemoryPoolDevicePtrItem {

    unsigned int   device_id;
    void         * ptr;

    gpuMemoryPoolDevicePtrItem( unsigned int device_id, void * ptr )
    {
      this->device_id = device_id;
      this->ptr       = ptr;
    }

    //This so it can be used in an STL map
    bool operator<( const gpuMemoryPoolDevicePtrItem & right ) const
    {
      if (this->device_id < right.device_id) {
        return true;
      }
      else if ((this->device_id == right.device_id) && (this->ptr < right.ptr)) {
        return true;
      }
      else {
        return false;
      }
    }
  };

  std::vector<gpuMemoryPoolDevicePtrItem> taskCudaMemoryPoolItems;
  std::queue<void*>                       taskHostMemoryPoolItems;

#endif

};
// end class DetailedTask

class DetailedTaskPriorityComparison {

public:

  bool operator()( DetailedTask *& ltask, DetailedTask *& rtask );
};

class DetailedTasks {

public:

  DetailedTasks(       SchedulerCommon * sc
               , const ProcessorGroup  * pg
               ,       DetailedTasks   * first
               , const TaskGraph       * taskgraph
               , const std::set<int>   & neighborhood_processors
               ,       bool              mustConsiderInternalDependencies = false
               );

  ~DetailedTasks();

  void add( DetailedTask * dtask );

  void makeDWKeyDatabase();

  void copyoutDWKeyDatabase( OnDemandDataWarehouseP dws )
  {
    dws->copyKeyDB(varKeyDB, levelKeyDB);
  }

  int numTasks() const
  {
    return (int)tasks_.size();
  }

  DetailedTask* getTask( int i )
  {
    return tasks_[i];
  }

  void assignMessageTags( int me );

  void initializeScrubs( std::vector<OnDemandDataWarehouseP> & dws, int dwmap[] );

  void possiblyCreateDependency(       DetailedTask     * from
                               ,       Task::Dependency * cmp
                               , const Patch            * fromPatch
                               ,       DetailedTask     * to
                               ,       Task::Dependency * req
                               , const Patch            * toPatch
                               ,       int                matl
                               , const IntVector        & low
                               , const IntVector        & high
                               ,       DepCommCond        cond
                               );

  DetailedTask* getOldDWSendTask( int proc );

  void logMemoryUse(       std::ostream  & out
                   ,       unsigned long & total
                   , const std::string   & tag
                   );

  void initTimestep();

  void computeLocalTasks( int me );

  int numLocalTasks() const
  {
    return (int)localtasks_.size();
  }

  DetailedTask* localTask( int idx )
  {
    return localtasks_[idx];
  }

  void emitEdges( ProblemSpecP edgesElement, int rank );

  DetailedTask* getNextInternalReadyTask();

  int numInternalReadyTasks();

  DetailedTask* getNextExternalReadyTask();

  int numExternalReadyTasks();

  void createScrubCounts();

  bool mustConsiderInternalDependencies()
  {
    return mustConsiderInternalDependencies_;
  }

  unsigned long getCurrentDependencyGeneration()
  {
    return currentDependencyGeneration_;
  }

  const TaskGraph* getTaskGraph() const
  {
    return taskgraph_;
  }

  void setScrubCount( const Task::Dependency                    * req
                    ,       int                                   matl
                    , const Patch                               * patch
                    ,       std::vector<OnDemandDataWarehouseP> & dws
                    );

  int getExtraCommunication()
  {
    return extraCommunication_;
  }

  friend std::ostream& operator<<( std::ostream & out, const Uintah::DetailedTask & task );

  friend std::ostream& operator<<( std::ostream & out, const Uintah::DetailedDep & task );

  ParticleExchangeVar& getParticleSends()
  {
    return particleSends_;
  }

  ParticleExchangeVar& getParticleRecvs()
  {
    return particleRecvs_;
  }

  void setTaskPriorityAlg( QueueAlg alg )
  {
    taskPriorityAlg_ = alg;
  }

  QueueAlg getTaskPriorityAlg()
  {
    return taskPriorityAlg_;
  }

#ifdef HAVE_CUDA

  void addVerifyDataTransferCompletion( DetailedTask * dtask );

  void addFinalizeDevicePreparation( DetailedTask * dtask );

  void addInitiallyReadyDeviceTask( DetailedTask * dtask );

  void addCompletionPendingDeviceTask( DetailedTask * dtask );

  void addFinalizeHostPreparation( DetailedTask * dtask );

  void addInitiallyReadyHostTask( DetailedTask * dtask );

  bool getNextVerifyDataTransferCompletionTaskIfAble( DetailedTask *& dtask );

  bool getNextFinalizeDevicePreparationTaskIfAble( DetailedTask *& dtask );

  bool getNextInitiallyReadyDeviceTaskIfAble( DetailedTask *& dtask );

  bool getNextCompletionPendingDeviceTaskIfAble( DetailedTask *& dtask );

  bool getNextFinalizeHostPreparationTaskIfAble( DetailedTask *& dtask );

  bool getNextInitiallyReadyHostTaskIfAble( DetailedTask *& dtask );

  void createInternalDependencyBatch(       DetailedTask     * from
                                    ,       Task::Dependency * comp
                                    , const Patch            * fromPatch
                                    ,       DetailedTask     * to
                                    ,       Task::Dependency * req
                                    , const Patch            * toPatch
                                    ,       int                matl
                                    , const IntVector        & low
                                    , const IntVector        & high
                                    ,       DepCommCond        cond
                                    );

  // helper of possiblyCreateDependency
  DetailedDep* findMatchingInternalDetailedDep(       DependencyBatch  * batch
                                              ,       DetailedTask     * toTask
                                              ,       Task::Dependency * req
                                              , const Patch            * fromPatch
                                              ,       int                matl
                                              ,       IntVector          low
                                              ,       IntVector          high
                                              ,       IntVector        & totalLow
                                              ,       IntVector        & totalHigh
                                              ,       DetailedDep      * &parent_dep
                                              );
#endif

protected:

  friend class DetailedTask;

  void internalDependenciesSatisfied( DetailedTask * dtask );

  SchedulerCommon* getSchedulerCommon()
  {
    return sc_;
  }

private:

  std::map<int, int>  sendoldmap_;
  ParticleExchangeVar particleSends_;
  ParticleExchangeVar particleRecvs_;

  void initializeBatches();

  void incrementDependencyGeneration();

  // helper of possiblyCreateDependency
  DetailedDep* findMatchingDetailedDep(       DependencyBatch  * batch
                                      ,       DetailedTask     * toTask
                                      ,       Task::Dependency * req
                                      , const Patch            * fromPatch
                                      ,       int                matl
                                      ,       IntVector          low
                                      ,       IntVector          high
                                      ,       IntVector        & totalLow
                                      ,       IntVector        & totalHigh
                                      ,       DetailedDep     *& parent_dep
                                      );

  void addScrubCount( const VarLabel * var
                    ,       int        matlindex
                    , const Patch    * patch
                    ,       int        dw
                    );

  bool getScrubCount( const VarLabel * var
                    , int              matlindex
                    , const Patch    * patch
                    , int              dw
                    , int            & count
                    );

  SchedulerCommon*      sc_;
  const ProcessorGroup* d_myworld;

  // store the first so we can share the scrubCountTable
  DetailedTasks*             first;
  std::vector<DetailedTask*> tasks_;
  KeyDatabase<Patch>         varKeyDB;
  KeyDatabase<Level>         levelKeyDB;

  const TaskGraph               * taskgraph_;
  Task                          * stask_;
  std::vector<DetailedTask*>      localtasks_;
  std::vector<DependencyBatch*>   batches_;
  DetailedDep                   * initreq_ { nullptr };

  // True for mixed scheduler which needs to keep track of internal dependencies.
  bool mustConsiderInternalDependencies_;

  // In the future, we may want to prioritize tasks for the MixedScheduler
  // to run.  I implemented this using topological sort order as the priority
  // but that probably isn't a good way to do unless you make it a breadth
  // first topological order.
  QueueAlg taskPriorityAlg_ { QueueAlg::MostMessages };

  using TaskQueue  = std::queue<DetailedTask*>;
  using TaskPQueue = std::priority_queue<DetailedTask*, std::vector<DetailedTask*>, DetailedTaskPriorityComparison>;

  TaskQueue  readyTasks_;
  TaskQueue  initiallyReadyTasks_;
  TaskPQueue mpiCompletedTasks_;

  // This "generation" number is to keep track of which InternalDependency
  // links have been satisfied in the current timestep and avoids the
  // need to traverse all InternalDependency links to reset values.
  unsigned long currentDependencyGeneration_ { 1 };

  // for logging purposes - how much extra comm is going on
  int extraCommunication_ { 0 };

  ScrubCountTable scrubCountTable_;

  // eliminate copy, assignment and move
  DetailedTasks(const DetailedTasks &)            = delete;
  DetailedTasks& operator=(const DetailedTasks &) = delete;
  DetailedTasks(DetailedTasks &&)                 = delete;
  DetailedTasks& operator=(DetailedTasks &&)      = delete;


#ifdef HAVE_CUDA
  TaskQueue  verifyDataTransferCompletionTasks_;   // Some or all ghost cells still need to be processed before a task is ready.
  TaskQueue  initiallyReadyDeviceTasks_;           // initially ready, h2d copies pending
  TaskQueue  finalizeDevicePreparationTasks_;      // h2d copies completed, need to mark gpu data as valid and copy gpu ghost cell data internally on device
  TaskQueue  completionPendingDeviceTasks_;        // execution and d2h copies pending
  TaskQueue  finalizeHostPreparationTasks_;        // d2h copies completed, need to mark cpu data as valid
  TaskQueue  initiallyReadyHostTasks_;             // initially ready cpu task, d2h copies pending
#endif

};
// end class DetailedTasks

std::ostream& operator<<( std::ostream & out, const Uintah::DetailedTask & task );

}  // End namespace Uintah

#endif // end CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

