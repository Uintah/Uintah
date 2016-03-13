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

#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Grid/Variables/ScrubItem.h>
#include <Core/Thread/CrowdMonitor.h>

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
#include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <list>
#include <queue>
#include <map>
#include <mutex>
#include <set>
#include <vector>


namespace Uintah {

class ProcessorGroup;
class DataWarehouse;
class DetailedTask;
class DetailedTasks;
class TaskGraph;
class SchedulerCommon;


  using ParticleExchangeVar = std::map<int, std::set<PSPatchMatlGhostRange> >;
  using ScrubCountTable     = SCIRun::FastHashTable<ScrubItem>;


  //_______________________________________________________________________________________________
  enum ProfileType {
      Normal
    , Fine
  };


  //_______________________________________________________________________________________________
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


  //_______________________________________________________________________________________________
  //
  //    DetailedDep
  //_______________________________________________________________________________________________
  //
  class DetailedDep {

  public:

    enum CommCondition {
        Always
      , FirstIteration
      , SubsequentIterations
    };

    DetailedDep(       DetailedDep * next
               ,       Task::Dependency   * comp
               ,       Task::Dependency   * req
               ,       DetailedTask       * toTask
               , const Patch              * fromPatch
               ,       int                  matl
               , const IntVector          & low
               , const IntVector          & high
               , CommCondition              cond
               )
      : next(next)
      , comp(comp)
      , req(req)
      , fromPatch(fromPatch)
      , low(low), high(high)
      , matl(matl)
      , condition(cond)
      , patchLow(low)
      , patchHigh(high)
    {
      ASSERT(Min(high - low, IntVector(1, 1, 1)) == IntVector(1, 1, 1));

      USE_IF_ASSERTS_ON( Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(), true); )

      ASSERT(fromPatch == 0 || (SCIRun::Min(low, fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())) ==
				     fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())));

      ASSERT(fromPatch == 0 || (SCIRun::Max(high, fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())) ==
				     fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())));

      toTasks.push_back(toTask);
    }


    // As an arbitrary convention, non-data dependency have a NULL fromPatch.
    // These types of dependency exist between a modifying task and any task
    // that requires the data (from ghost cells in particular) before it is
    // modified preventing the possibility of modifying data while it is being
    // used.
    bool isNonDataDependency() const { return (fromPatch == nullptr); }

    DetailedDep              * next;
    Task::Dependency         * comp;
    Task::Dependency         * req;
    std::list<DetailedTask*>   toTasks;
    const Patch              * fromPatch;
    IntVector                  low;
    IntVector                  high;
    int                        matl;

    // this is to satisfy a need created by the DynamicLoadBalancer.  To keep it unrestricted on when it can perform, and
    // to avoid a costly second recompile on the next timestep, we add a comm condition which will send/recv data based
    // on whether some condition is met at run time - in this case whether it is the first execution or not.
    CommCondition condition;

    // for SmallMessages - if we don't copy the complete patch, we need to know the range so we can store all segments properly
    IntVector patchLow;
    IntVector patchHigh;

    // eliminate copy, assignment and move
    DetailedDep( const DetailedDep & )            = delete;
    DetailedDep& operator=( const DetailedDep & ) = delete;
    DetailedDep( DetailedDep && )                 = delete;
    DetailedDep& operator=( DetailedDep && )      = delete;


  }; // DetailedDep
  //_______________________________________________________________________________________________


  //_______________________________________________________________________________________________
  //
  //    DependencyBatch
  //_______________________________________________________________________________________________
  //
  class DependencyBatch {

  public:

    DependencyBatch( int           to
                   ,  DetailedTask* fromTask
                   ,  DetailedTask* toTask
                   )
      : comp_next(0)
      , fromTask(fromTask)
      , head(0), messageTag(-1)
      , to(to), received_(false)
      , madeMPIRequest_(false)
    {
      toTasks.push_back(toTask);
    }

    ~DependencyBatch();

    // The first thread calling this will return true, all others
    // will return false.
    bool makeMPIRequest();

    // Tells this batch that it has actually been received and
    // awakens anybody blocked in makeMPIRequest().
    void received( const ProcessorGroup * pg );

    bool wasReceived() { return received_; }

    // Initialize receiving information for makeMPIRequest() and received()
    // so that it can receive again.
    void reset();

    //Add invalid variables to the dependency batch.  These variables will be marked
    //as valid when MPI completes.
    void addVar( Variable* var ) { toVars.push_back(var); }

    void addReceiveListener( int mpiSignal );


    DependencyBatch          * comp_next;
    DetailedTask             * fromTask;
    std::list<DetailedTask*>   toTasks;
    DetailedDep              * head;
    int                        messageTag;
    int                        to;

    //scratch pad to store wait times for debugging
    static std::map<std::string,double> waittimes;

  private:

    volatile bool  received_;
    volatile bool  madeMPIRequest_;
    std::mutex     lock_;
    std::set<int>  receiveListeners_;

    // eliminate copy, assignment and move
    DependencyBatch( const DependencyBatch & )            = delete;
    DependencyBatch& operator=( const DependencyBatch & ) = delete;
    DependencyBatch( DependencyBatch && )                 = delete;
    DependencyBatch& operator=( DependencyBatch && )      = delete;

    std::vector<Variable*> toVars;

  }; // DependencyBatch
  //_______________________________________________________________________________________________



  //_______________________________________________________________________________________________
  //
  //    InternalDependency
  //_______________________________________________________________________________________________
  //
  struct InternalDependency {

    InternalDependency(       DetailedTask* prerequisiteTask
                      ,        DetailedTask* dependentTask
                      , const VarLabel*     var
                      ,        long          satisfiedGeneration
                      )
      : prerequisiteTask(prerequisiteTask)
      , dependentTask(dependentTask)
      , satisfiedGeneration(satisfiedGeneration)
    {
      addVarLabel(var);
    }

    void addVarLabel( const VarLabel* var ) { vars.insert(var); }

    DetailedTask * prerequisiteTask;
    DetailedTask * dependentTask;

    std::set<const VarLabel*, VarLabel::Compare>  vars;
    unsigned long                                 satisfiedGeneration;

  }; // InternalDependency
  //_______________________________________________________________________________________________



#ifdef HAVE_CUDA
  struct TaskGpuDataWarehouses{
    public:
    GPUDataWarehouse* TaskGpuDW[2];
  };

#endif



  //_______________________________________________________________________________________________
  //
  //    DetailedTask
  //_______________________________________________________________________________________________
  //
  class DetailedTask {

  public:

    DetailedTask(       Task           * task
                , const PatchSubset    * patches
                , const MaterialSubset * matls
                ,       DetailedTasks  * taskGroup
                );

    ~DetailedTask();

    void setProfileType( ProfileType type ) { d_profileType=type; }

    ProfileType getProfileType() { return d_profileType; }

    void doit( const ProcessorGroup                      * pg
             ,       std::vector<OnDemandDataWarehouseP> & oddws
             ,       std::vector<DataWarehouseP>         & dws
             ,       Task::CallBackEvent                   event = Task::CPU
             );

    // Called after doit and mpi data sent (packed in buffers) finishes.
    // Handles internal dependencies and scrubbing.
    // Called after doit finishes.
    void done( std::vector<OnDemandDataWarehouseP>& dws );

    std::string getName() const;

    const Task* getTask() const { return task; }

    const PatchSubset* getPatches() const { return patches; }

    const MaterialSubset* getMaterials() const { return matls; }

    void assignResource(int idx) { resourceIndex = idx; }

    int getAssignedResourceIndex() const { return resourceIndex; }

    void assignStaticOrder( int i ) { staticOrder = i; }

    int getStaticOrder() const { return staticOrder; }

    DetailedTasks* getTaskGroup() const { return taskGroup; }

    std::map<DependencyBatch*, DependencyBatch*>& getRequires() { return reqs; }

    std::map<DependencyBatch*, DependencyBatch*>& getInternalRequires() { return internal_reqs; }

    DependencyBatch* getComputes() const { return comp_head; }

    DependencyBatch* getInternalComputes() const { return internal_comp_head; }

    void findRequiringTasks( const VarLabel* var, std::list<DetailedTask*>& requiringTasks );

    void emitEdges( ProblemSpecP edgesElement );

    bool addInternalRequires( DependencyBatch*);

    void addInternalComputes( DependencyBatch* );

    bool addRequires( DependencyBatch* );

    void addComputes( DependencyBatch* );

    void addInternalDependency( DetailedTask* prerequisiteTask, const VarLabel* var );

    // external dependencies will count how many messages this task
    // is waiting for.  When it hits 0, we can add it to the
    // DetailedTasks::mpiCompletedTasks list.
    void resetDependencyCounts();

    bool isInitiated() const { return initiated_; }

    void markInitiated() { initiated_ = true; }

    void incrementExternalDepCount() { externalDependencyCount_++; }

    void decrementExternalDepCount() { externalDependencyCount_--; }

    void checkExternalDepCount();

    int getExternalDepCount() { return externalDependencyCount_; }

    bool areInternalDependenciesSatisfied() { return (numPendingInternalDependencies == 0); }


#ifdef HAVE_CUDA

    void assignDevice (unsigned int device);

    //unsigned int getDeviceNum() const;

    //Most tasks will only run on one device.
    //But some, such as the data archiver task or send old data could run on multiple devices.
    //This is not a good idea.  A task should only run on one device.  But the capability for a task
    //to run on multiple nodes exists.
    std::set<unsigned int> getDeviceNums() const;
    std::map<unsigned int, TaskGpuDataWarehouses> TaskGpuDWs;


    //bool queryCUDAStreamCompletion();

    //void setCUDAStream(cudaStream_t* s);

    void setCUDAStream(unsigned int deviceNum, cudaStream_t* s);

    void clearCUDAStreams();

    bool checkCUDAStreamDone() const;

    bool checkCUDAStreamDone(unsigned int deviceNum) const;

    bool checkAllCUDAStreamsDone() const;



    void setTaskGpuDataWarehouse(unsigned int deviceNum, Task::WhichDW DW, GPUDataWarehouse* TaskDW);

    GPUDataWarehouse* getTaskGpuDataWarehouse(unsigned int deviceNum, Task::WhichDW DW);

    void deleteTaskGpuDataWarehouses();

    cudaStream_t* getCUDAStream(unsigned int deviceNum) const;

    DeviceGridVariables& getDeviceVars() { return deviceVars; }
    DeviceGridVariables& getTaskVars() { return taskVars; }
    DeviceGhostCells& getGhostVars() { return ghostVars; }
    DeviceGridVariables& getVarsToBeGhostReady() { return varsToBeGhostReady; }
    DeviceGridVariables& getVarsBeingCopiedByTask() { return varsBeingCopiedByTask; }
    void clearPreparationCollections();


#endif


  protected:

    friend class TaskGraph;

  private:

    // called by done()
    void scrub( std::vector<OnDemandDataWarehouseP>& );

    // eliminate copy, assignment and move
    DetailedTask( const DetailedTask & )            = delete;
    DetailedTask& operator=( const DetailedTask & ) = delete;
    DetailedTask( DetailedTask && )                 = delete;
    DetailedTask& operator=( DetailedTask && )      = delete;

    Task                                         * task;
    const PatchSubset                            *  patches;
    const MaterialSubset                         * matls;
    std::map<DependencyBatch*, DependencyBatch*>   reqs;
    std::map<DependencyBatch*, DependencyBatch*>   internal_reqs;
    DependencyBatch                              * comp_head;
    DependencyBatch                              *  internal_comp_head;
    DetailedTasks                                * taskGroup;

    bool initiated_;
    bool externallyReady_;
    int  externalDependencyCount_;

    mutable std::string name_; // doesn't get set until getName() is called the first time.

    // Called when prerequisite tasks (dependencies) call done.
    void dependencySatisfied( InternalDependency* dep );

    // Internal dependencies are dependencies within the same process.
    std::list<InternalDependency> internalDependencies;

    // internalDependents will point to InternalDependency's in the
    // internalDependencies list of the requiring DetailedTasks.
    std::map<DetailedTask*, InternalDependency*> internalDependents;

    unsigned long   numPendingInternalDependencies;
    std::mutex      internalDependencyLock;

    int resourceIndex;
    int staticOrder;

    DetailedTask( const Task& );
    DetailedTask& operator=( const Task& );

    // specifies the type of task this is:
    //   * normal executes on either the patches cells or the patches coarse cells
    //   * fine executes on the patches fine cells (for example coarsening)

    bool operator<( const DetailedTask& other );

    ProfileType d_profileType;

#ifdef HAVE_CUDA
    bool deviceExternallyReady_;
    bool completed_;
    unsigned int  deviceNum_;
    std::set <unsigned int> deviceNums_;
    //cudaStream_t*   d_cudaStream;
    std::map <unsigned int, cudaStream_t*> d_cudaStreams;

    //Store information about each set of grid variables.
    //This will help later when we figure out the best way to store data into the GPU.
    //It may be stored contiguously.  It may handle material data.  It just helps to gather it all up
    //into a collection prior to copying data.
    DeviceGridVariables deviceVars; //Holds variables that will need to be copied into the GPU
    DeviceGridVariables taskVars;   //Holds variables that will be needed for a GPU task (a Task DW has a snapshot of
                                    //all important pointer info from the host-side GPU DW)
    DeviceGhostCells ghostVars;     //Holds ghost cell meta data copy information

    DeviceGridVariables varsToBeGhostReady; //Holds a list of vars this task is managing to ensure their ghost cells will be ready.
                                            //This means this task is the exlusive ghost cell gatherer and ghost cell validator for any
                                            //label/patch/matl/level vars it has listed in here
                                            //But it is NOT the exclusive copier.  Because some ghost cells from one patch may be used by
                                            //two or more destination patches.  We only want to copy ghost cells once.

    DeviceGridVariables varsBeingCopiedByTask;  //Holds a list of the vars that this task is actually copying into the GPU.
#endif


  }; // end class DetailedTask



  //_______________________________________________________________________________________________
  //
  //    DetailedTaskPriorityComparison
  //_______________________________________________________________________________________________
  //
  class DetailedTaskPriorityComparison
  {
    public:

      bool operator()( DetailedTask*& ltask, DetailedTask*& rtask );
  };
  //_______________________________________________________________________________________________




  //_______________________________________________________________________________________________
  //
  //    DetailedTasks
  //_______________________________________________________________________________________________
  //
  class DetailedTasks {

  public:

    DetailedTasks(        SchedulerCommon * sc
                 ,  const ProcessorGroup  * pg
                 ,        DetailedTasks   * first
                 ,  const TaskGraph       * taskgraph
                 ,  const std::set<int>   & neighborhood_processors
                 ,        bool              mustConsiderInternalDependencies = false
                 );

    ~DetailedTasks();

    void add( DetailedTask* task );

    void makeDWKeyDatabase();

    void copyoutDWKeyDatabase( OnDemandDataWarehouseP dws ) { dws->copyKeyDB(varKeyDB, levelKeyDB); }

    int numTasks() const { return (int)tasks_.size(); }

    DetailedTask* getTask( int i ) { return tasks_[i]; }

    void assignMessageTags( int me );

    void initializeScrubs( std::vector<OnDemandDataWarehouseP>& dws, int dwmap[] );

    void possiblyCreateDependency(        DetailedTask                       * from
                                 ,        Task::Dependency                   * comp
                                 ,  const Patch                              * fromPatch
                                 ,        DetailedTask                       * to
                                 ,        Task::Dependency                   * req
                                 ,  const Patch                              * toPatch
                                 ,        int                                  matl
                                 ,  const IntVector                          & low
                                 ,  const IntVector                          & high
                                 ,        DetailedDep::CommCondition           cond
                                 );

    DetailedTask* getOldDWSendTask(int proc);

    void logMemoryUse( std::ostream& out, unsigned long& total, const std::string& tag );

    void initTimestep();

    void computeLocalTasks( int me );

    int numLocalTasks() const { return (int)localtasks_.size(); }

    DetailedTask* localTask( int idx ) { return localtasks_[idx]; }

    void emitEdges( ProblemSpecP edgesElement, int rank );

    DetailedTask* getNextInternalReadyTask();

    int numInternalReadyTasks();

    DetailedTask* getNextExternalReadyTask();

    int numExternalReadyTasks();

    void createScrubCounts();

    bool mustConsiderInternalDependencies() { return mustConsiderInternalDependencies_; }

    unsigned long getCurrentDependencyGeneration() { return currentDependencyGeneration_; }

    const TaskGraph* getTaskGraph() const { return taskgraph_; }

    void setScrubCount( const Task::Dependency                     * req
                      ,        int                                   matl
                      ,  const Patch                               * patch
                      ,        std::vector<OnDemandDataWarehouseP> & dws
                      );

    int getExtraCommunication() { return extraCommunication_; }

    ParticleExchangeVar& getParticleSends() { return particleSends_; }

    ParticleExchangeVar& getParticleRecvs() { return particleRecvs_; }

    void setTaskPriorityAlg( QueueAlg alg ) { taskPriorityAlg_=alg; }

    QueueAlg getTaskPriorityAlg() { return taskPriorityAlg_; }


#ifdef HAVE_CUDA
    void addVerifyDataTransferCompletion(DetailedTask* dtask);
    void addFinalizeDevicePreparation(DetailedTask* dtask);
    void addInitiallyReadyDeviceTask( DetailedTask* dtask );
    void addCompletionPendingDeviceTask( DetailedTask* dtask );
    void addFinalizeHostPreparation(DetailedTask* dtask);
    void addInitiallyReadyHostTask(DetailedTask* dtask);

    DetailedTask* getNextVerifyDataTransferCompletionTask();
    DetailedTask* getNextFinalizeDevicePreparationTask();
    DetailedTask* getNextInitiallyReadyDeviceTask();
    DetailedTask* getNextCompletionPendingDeviceTask();
    DetailedTask* getNextFinalizeHostPreparationTask();
    DetailedTask* getNextInitiallyReadyHostTask();

    DetailedTask* peekNextVerifyDataTransferCompletionTask();
    DetailedTask* peekNextFinalizeDevicePreparationTask();
    DetailedTask* peekNextInitiallyReadyDeviceTask();
    DetailedTask* peekNextCompletionPendingDeviceTask();
    DetailedTask* peekNextFinalizeHostPreparationTask();
    DetailedTask* peekNextInitiallyReadyHostTask();

    int numVerifyDataTransferCompletion() { return verifyDataTransferCompletionTasks_.size(); }
    int numFinalizeDevicePreparation() { return finalizeDevicePreparationTasks_.size(); }
    int numInitiallyReadyDeviceTasks() { return initiallyReadyDeviceTasks_.size(); }
    int numCompletionPendingDeviceTasks() { return completionPendingDeviceTasks_.size(); }
    int numFinalizeHostPreparation() { return finalizeHostPreparationTasks_.size(); }
    int numInitiallyReadyHostTasks() { return initiallyReadyHostTasks_.size(); }

    void createInternalDependencyBatch(DetailedTask* from,
                                   Task::Dependency* comp,
                                   const Patch* fromPatch,
                                   DetailedTask* to,
                                   Task::Dependency* req,
                                   const Patch *toPatch,
                                   int matl,
                                   const IntVector& low,
                                   const IntVector& high,
                                   DetailedDep::CommCondition cond);
    // helper of possiblyCreateDependency
    DetailedDep* findMatchingInternalDetailedDep(DependencyBatch* batch, DetailedTask* toTask, Task::Dependency* req,
                                         const Patch* fromPatch, int matl, IntVector low, IntVector high,
                                         IntVector& totalLow, IntVector& totalHigh, DetailedDep* &parent_dep);
#endif

  protected:

    friend class DetailedTask;

    void internalDependenciesSatisfied( DetailedTask* task );

    SchedulerCommon* getSchedulerCommon() { return sc_; }

  private:

    // eliminate copy, assignment and move
    DetailedTasks( const DetailedTasks & )            = delete;
    DetailedTasks& operator=( const DetailedTasks & ) = delete;
    DetailedTasks( DetailedTasks && )                 = delete;
    DetailedTasks& operator=( DetailedTasks && )      = delete;

    void initializeBatches();

    void incrementDependencyGeneration();

    // helper of possiblyCreateDependency
    DetailedDep* findMatchingDetailedDep(       DependencyBatch    * batch
                                        ,       DetailedTask       * toTask
                                        ,       Task::Dependency   * req
                                        , const Patch              * fromPatch
                                        ,       int                   matl
                                        ,       IntVector             low
                                        ,       IntVector             high
                                        ,       IntVector           & totalLow
                                        ,       IntVector           & totalHigh
                                        ,       DetailedDep        *& parent_dep
                                        );


    void addScrubCount( const VarLabel * var
                      ,       int        matlindex
                      , const Patch    * patch
                      ,       int        dw
                      );


    bool getScrubCount( const VarLabel * var
                      ,       int        matlindex
                      , const Patch    * patch
                      ,       int        dw
                      ,       int      & count
                      );

    std::map<int,int>   sendoldmap_;
    ParticleExchangeVar particleSends_;
    ParticleExchangeVar particleRecvs_;

    SchedulerCommon             * sc_;
    const ProcessorGroup        * d_myworld;
    // store the first so we can share the scrubCountTable
    DetailedTasks               * first;
    std::vector<DetailedTask*>    tasks_;
    KeyDatabase<Patch>            varKeyDB;
    KeyDatabase<Level>            levelKeyDB;

    // TODO - FIXME: Figure out why this was commented out long ago - APH 02/12/16
    #if 0
        std::vector<DetailedReq*>  initreqs_;
    #endif

    const TaskGraph               * taskgraph_;
    Task                          * stask_;
    std::vector<DetailedTask*>      localtasks_;
    std::vector<DependencyBatch*>   batches_;
    DetailedDep                   * initreq_;

    // True for mixed scheduler which needs to keep track of internal dependencies.
    bool mustConsiderInternalDependencies_;

    // In the future, we may want to prioritize tasks for the MixedScheduler
    // to run.  I implemented this using topological sort order as the priority
    // but that probably isn't a good way to do unless you make it a breadth
    // first topological order.
    QueueAlg taskPriorityAlg_;

    using TaskQueue  = std::queue<DetailedTask*>;
    using TaskPQueue = std::priority_queue<DetailedTask*, std::vector<DetailedTask*>, DetailedTaskPriorityComparison>;

    TaskQueue   readyTasks_;
    TaskQueue   initiallyReadyTasks_;
    TaskPQueue  mpiCompletedTasks_;

    // This "generation" number is to keep track of which InternalDependency
    // links have been satisfied in the current timestep and avoids the
    // need to traverse all InternalDependency links to reset values.
    unsigned long currentDependencyGeneration_;

    // for logging purposes - how much extra comm is going on
    int extraCommunication_;

    std::mutex  readyQueueLock_;
    std::mutex  mpiCompletedQueueLock_;

    ScrubCountTable scrubCountTable_;


#ifdef HAVE_CUDA
    TaskPQueue            verifyDataTransferCompletionTasks_;    // Some or all ghost cells still need to be processed before a task is ready.
    TaskPQueue            initiallyReadyDeviceTasks_;       // initially ready, h2d copies pending
    TaskPQueue            finalizeDevicePreparationTasks_;  // h2d copies completed, need to mark gpu data as valid and copy gpu ghost cell data internally on device
    TaskPQueue            completionPendingDeviceTasks_;    // execution and d2h copies pending
    TaskPQueue            finalizeHostPreparationTasks_;    // d2h copies completed, need to mark cpu data as valid
    TaskPQueue            initiallyReadyHostTasks_;         // initially ready cpu task, d2h copies pending

    mutable CrowdMonitor  deviceVerifyDataTransferCompletionQueueLock_;
    mutable CrowdMonitor  deviceFinalizePreparationQueueLock_;
    mutable CrowdMonitor  deviceReadyQueueLock_;
    mutable CrowdMonitor  deviceCompletedQueueLock_;
    mutable CrowdMonitor  hostFinalizePreparationQueueLock_;
    mutable CrowdMonitor  hostReadyQueueLock_;
#endif


    friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );
    friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDep& task );

  }; // DetailedTasks
  //_______________________________________________________________________________________________

  std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );
  std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDep& task );

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

