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
#include <Core/Malloc/AllocatorTags.hpp>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Mutex.h>

#include <queue>

#include <sci_defs/cuda_defs.h>


namespace Uintah {


  class TaskGraph;
  class SchedulerCommon;


  using ParticleExchangeVar = std::map<int, std::set<PSPatchMatlGhostRange> >;
  using ScrubCountTable     = SCIRun::FastHashTable<ScrubItem>;
  
  using TaskQueue = Lockfree::CircularPool<   DetailedTask*
                                            , Lockfree::ENABLE_SIZE         // size model
                                            , Lockfree::EXCLUSIVE_INSTANCE // usage model
                                            , Uintah::MallocAllocator      // allocator
                                            , Uintah::MallocAllocator      // size_type allocator
                                          >;


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




  //_______________________________________________________________________________________________
  //
  //    DetailedDependency
  //_______________________________________________________________________________________________
  //
  class DetailedDependency {

  public:

    enum CommCondition {
        Always
      , FirstIteration
      , SubsequentIterations
    };

    DetailedDependency(       DetailedDependency * next
                      ,       Task::Dependency   * comp
                      ,       Task::Dependency   * req
                      ,       DetailedTask       * toTask
                      , const Patch              * fromPatch
                      ,       int                  matl
                      , const IntVector          & low
                      , const IntVector          & high
                      , CommCondition              cond
                      )
      : m_next{ next }
      , m_comp{ comp }
      , m_req{ req }
      , m_from_patch{ fromPatch }
      , m_low{ low }
      , m_high{ high }
      , m_matl{ matl }
      , m_comm_condition{ cond }
      , m_patch_low{ low }
      , m_patch_high{ high }
    {
      ASSERT(SCIRun::Min(high - low, IntVector(1, 1, 1)) == IntVector(1, 1, 1));

      USE_IF_ASSERTS_ON( Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(), true); )

      ASSERT(fromPatch == 0 || (SCIRun::Min(low, fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())) ==
				     fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())));

      ASSERT(fromPatch == 0 || (SCIRun::Max(high, fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())) ==
				     fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())));

      m_to_tasks.push_back(toTask);
    }


    // As an arbitrary convention, non-data dependency have a NULL fromPatch.
    // These types of dependency exist between a modifying task and any task
    // that requires the data (from ghost cells in particular) before it is
    // modified preventing the possibility of modifying data while it is being
    // used.
    bool isNonDataDependency() const { return (m_from_patch == nullptr); }

    DetailedDependency       * m_next;
    Task::Dependency         * m_comp;
    Task::Dependency         * m_req;
    std::list<DetailedTask*>   m_to_tasks;
    const Patch              * m_from_patch;
    IntVector                  m_low;
    IntVector                  m_high;
    int                        m_matl;

    // this is to satisfy a need created by the DynamicLoadBalancer.  To keep it unrestricted on when it can perform, and 
    // to avoid a costly second recompile on the next timestep, we add a comm condition which will send/recv data based
    // on whether some condition is met at run time - in this case whether it is the first execution or not.
    CommCondition m_comm_condition;

    // for SmallMessages - if we don't copy the complete patch, we need to know the range so we can store all segments properly
    IntVector m_patch_low;
    IntVector m_patch_high;

    // eliminate copy, assignment and move
    DetailedDependency( const DetailedDependency & )            = delete;
    DetailedDependency& operator=( const DetailedDependency & ) = delete;
    DetailedDependency( DetailedDependency && )                 = delete;
    DetailedDependency& operator=( DetailedDependency && )      = delete;

  }; // DetailedDependency
  //_______________________________________________________________________________________________




  //_______________________________________________________________________________________________
  //
  //    DependencyBatch
  //_______________________________________________________________________________________________
  //
  class DependencyBatch {

  public:

    DependencyBatch( int            to
                   , DetailedTask * fromTask
                   , DetailedTask * toTask
                   )
      : m_comp_next{ 0 }
      , m_from_task{ fromTask }
      , m_head{ 0 }
      , m_message_tag{ -1 }
      , m_to_rank{ to }
      , m_received{ false }
      , m_made_mpi_request{ false }
      , m_lock{ 0 }
    {
      m_to_tasks.push_back(toTask);
    }

    ~DependencyBatch();

    // The first thread calling this will return true, all others
    // will return false.
    bool makeMPIRequest();

    // Tells this batch that it has actually been received and
    // awakens anybody blocked in makeMPIRequest().
    void received( const ProcessorGroup * pg );

    bool wasReceived() { return m_received; }

    // Initialize receiving information for makeMPIRequest() and received()
    // so that it can receive again.
    void reset();

    //Add invalid variables to the dependency batch.  These variables will be marked
    //as valid when MPI completes. 
    void addVar( Variable* var ) { m_to_variables.push_back(var); }

    // TODO - FIXME: Figure out why this was commented out long ago - APH 02/12/16
//    DependencyBatch          * req_next;
    
    DependencyBatch          * m_comp_next;
    DetailedTask             * m_from_task;
    std::list<DetailedTask*>   m_to_tasks;
    DetailedDependency       * m_head;
    int                        m_message_tag;
    int                        m_to_rank;

    //scratch pad to store wait times for debugging
    static std::map<std::string, double> s_wait_times;


  private:

    volatile bool            m_received;
    volatile bool            m_made_mpi_request;
    SCIRun::Mutex          * m_lock;
    std::vector<Variable*>   m_to_variables;

    // eliminate copy, assignment and move
    DependencyBatch( const DependencyBatch & )            = delete;
    DependencyBatch& operator=( const DependencyBatch & ) = delete;
    DependencyBatch( DependencyBatch && )                 = delete;
    DependencyBatch& operator=( DependencyBatch && )      = delete;

  }; // DependencyBatch
  //_______________________________________________________________________________________________




  //_______________________________________________________________________________________________
  //
  //    InternalDependency
  //_______________________________________________________________________________________________
  //
  struct InternalDependency {

    InternalDependency(       DetailedTask  * prerequisiteTask
                      ,       DetailedTask  * dependentTask
                      , const VarLabel      * var
                      ,       unsigned long   satisfiedGeneration
                      )
      : m_prerequisite_task{ prerequisiteTask }
      , m_dependent_task{ dependentTask }
      , m_satisfied_generation{ satisfiedGeneration }
    {
      addVarLabel(var);
    }

    void addVarLabel( const VarLabel* var ) { m_variables.insert(var); }
    
    DetailedTask * m_prerequisite_task;
    DetailedTask * m_dependent_task;

    std::set<const VarLabel*, VarLabel::Compare> m_variables;
    unsigned long                                m_satisfied_generation;

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

    DetailedTask(      Task            * task
                , const PatchSubset    * patches
                , const MaterialSubset * matls
                ,       DetailedTasks  * taskGroup
                );

    ~DetailedTask();
   
    void setProfileType( ProfileType type ) { m_profile_type=type; }

    ProfileType getProfileType() { return m_profile_type; }


    void doit( const ProcessorGroup                       * pg
             ,        std::vector<OnDemandDataWarehouseP> & oddws
             ,        std::vector<DataWarehouseP>         & dws
             ,        Task::CallBackEvent                   event = Task::CPU
             );

    // Called after doit and mpi data sent (packed in buffers) finishes.
    // Handles internal dependencies and scrubbing.
    // Called after doit finishes.
    void done(std::vector<OnDemandDataWarehouseP>& dws);

    std::string getName() const;
    
    const Task* getTask() const { return m_task; }

    const PatchSubset* getPatches() const { return m_patches; }

    const MaterialSubset* getMaterials() const { return m_matls; }

    void assignResource(int idx) { m_resource_index = idx; }

    int getAssignedResourceIndex() const { return m_resource_index; }
    
    void assignStaticOrder( int i ) { m_static_order = i; }

    int getStaticOrder() const { return m_static_order; }
 
    DetailedTasks* getTaskGroup() const { return m_task_group; }

    std::map<DependencyBatch*, DependencyBatch*>& getRequires() { return m_requires; }

    std::map<DependencyBatch*, DependencyBatch*>& getInternalRequires() { return m_internal_reqs; }  
  
    DependencyBatch* getComputes() const { return m_comp_head; }

    DependencyBatch* getInternalComputes() const { return m_internal_comp_head; }
    
    void findRequiringTasks( const VarLabel* var, std::list<DetailedTask*>& requiringTasks );

    void emitEdges( ProblemSpecP edgesElement );

    bool addInternalRequires( DependencyBatch* );

    void addInternalComputes( DependencyBatch* );

    bool addRequires( DependencyBatch* );

    void addComputes( DependencyBatch* );

    void addInternalDependency( DetailedTask* prerequisiteTask, const VarLabel* var );

    void resetDependencyCounts();

    void markInitiated() { m_initiated = true; }

    void incrementExternalDepCount() { m_external_dependency_count++; }

    void decrementExternalDepCount() { m_external_dependency_count--; }

    // external dependencies will count how many messages this task
    // is waiting for.  When it hits 0, we can add it to the
    // DetailedTasks::mpiCompletedTasks list.
    void checkExternalDepCount();

    int getExternalDepCount() { return m_external_dependency_count; }

    bool areInternalDependenciesSatisfied() { return (m_num_pending_internal_dependencies == 0); }


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
#endif


  protected:

    friend class TaskGraph;

  private:

    // eliminate copy, assignment and move
    DetailedTask( const DetailedTask & )            = delete;
    DetailedTask& operator=( const DetailedTask & ) = delete;
    DetailedTask( DetailedTask && )                 = delete;
    DetailedTask& operator=( DetailedTask && )      = delete;

    // called by done()
    void scrub( std::vector<OnDemandDataWarehouseP>& );

    // Called when prerequisite tasks (dependencies) call done.
    void dependencySatisfied( InternalDependency* dep );

    Task                                         * m_task;
    const PatchSubset                            * m_patches;
    const MaterialSubset                         * m_matls;
    std::map<DependencyBatch*, DependencyBatch*>   m_requires;
    std::map<DependencyBatch*, DependencyBatch*>   m_internal_reqs;
    DependencyBatch                              * m_comp_head;
    DependencyBatch                              * m_internal_comp_head;
    DetailedTasks                                * m_task_group;

    bool m_initiated;
    bool m_externally_ready;
    int  m_external_dependency_count;

    mutable std::string m_name; // doesn't get set until getName() is called the first time.


    // Internal dependencies are dependencies within the same process.
    std::list<InternalDependency> m_internal_dependencies;
    
    // internalDependents will point to InternalDependency's in the
    // internalDependencies list of the requiring DetailedTasks.
    std::map<DetailedTask*, InternalDependency*> m_internal_dependents;
    
    unsigned long  m_num_pending_internal_dependencies;
    SCIRun::Mutex  m_internal_dependency_lock;
    
    int m_resource_index;
    int m_static_order;
    
    // specifies the type of task this is:
    //   * normal executes on either the patches cells or the patches coarse cells
    //   * fine executes on the patches fine cells (for example coarsening)
    
    bool operator<( const DetailedTask& other );
    
    ProfileType m_profile_type;


#ifdef HAVE_CUDA
    bool deviceExternallyReady_;
    bool completed_;
    unsigned int  deviceNum_;
    std::set <unsigned int> deviceNums_;
    //cudaStream_t*   d_cudaStream;
    std::map <unsigned int, cudaStream_t*> d_cudaStreams;
#endif


  }; // end class DetailedTask
  //_______________________________________________________________________________________________



  
#ifdef HAVE_CUDA

  struct varTuple {
     std::string     label;
     int             matlIndx;
     int             levelIndx;
     int             patch;
     int             dataWarehouse;
     IntVector       sharedLowCoordinates;
     IntVector       sharedHighCoordinates;

     varTuple(std::string label, int matlIndx, int levelIndx, int patch, int dataWarehouse, IntVector sharedLowCoordinates, IntVector sharedHighCoordinates) {
       this->label = label;
       this->matlIndx = matlIndx;
       this->levelIndx = levelIndx;
       this->patch = patch;
       this->dataWarehouse = dataWarehouse;
       this->sharedLowCoordinates = sharedLowCoordinates;
       this->sharedHighCoordinates = sharedHighCoordinates;
     }
     //This is so it can be used in an STL map
     bool operator<(const varTuple& right) const {
       if (this->label < right.label) {
         return true;
       } else if (this->label == right.label && (this->matlIndx < right.matlIndx)) {
         return true;
       } else if (this->label == right.label && (this->matlIndx == right.matlIndx)
                  && (this->levelIndx < right.levelIndx)) {
         return true;
       } else if (this->label == right.label && (this->matlIndx == right.matlIndx)
                  && (this->levelIndx == right.levelIndx) && (this->patch < right.patch)) {
         return true;
       } else if (this->label == right.label && (this->matlIndx == right.matlIndx)
                  && (this->levelIndx == right.levelIndx)
                  && (this->patch == right.patch)
                  && (this->dataWarehouse < right.dataWarehouse)) {
         return true;
       } else if (this->label == right.label && (this->matlIndx == right.matlIndx)
           && (this->levelIndx == right.levelIndx)
           && (this->patch == right.patch)
           && (this->dataWarehouse == right.dataWarehouse)
           && (this->sharedLowCoordinates < right.sharedLowCoordinates)) {
         return true;
       } else if (this->label == right.label && (this->matlIndx == right.matlIndx)
           && (this->levelIndx == right.levelIndx)
           && (this->patch == right.patch)
           && (this->dataWarehouse == right.dataWarehouse)
           && (this->sharedLowCoordinates == right.sharedLowCoordinates)
           && (this->sharedHighCoordinates < right.sharedHighCoordinates)) {
         return true;
       } else {
         return false;
       }
     }
   };

  class CopyDependenciesGpu
  {
    public:
      void addVar(std::string label, int matlIndx, int levelIndx, int patch, int dataWarehouse, IntVector sharedLowCoordinates, IntVector sharedHighCoordinates){
        varTuple temp(label, matlIndx, levelIndx, patch, dataWarehouse, sharedLowCoordinates, sharedHighCoordinates);
        varsBeingCopied.push_back(temp);
      }
    private:
      //uniquely identify a variable by a tuple of label/patch/material/level/dw/low/high
      cudaStream_t* stream;
      unsigned int device;
      std::vector<varTuple> varsBeingCopied;
  }; // end class CopyDependencies

#endif





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
    
    void copyoutDWKeyDatabase( OnDemandDataWarehouseP dws ) { dws->copyKeyDB(m_var_key_DB, m_level_key_DB); }

    int numTasks() const { return (int)m_tasks.size(); }

    DetailedTask* getTask( int i ) { return m_tasks[i]; }

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
                                 ,        DetailedDependency::CommCondition    cond
                                 );

    DetailedTask* getOldDWSendTask(int proc);

    void logMemoryUse( std::ostream& out, unsigned long& total, const std::string& tag );

    void initTimestep();
    
    void computeLocalTasks( int me );

    int numLocalTasks() const { return (int)m_local_tasks.size(); }

    DetailedTask* localTask( int idx ) { return m_local_tasks[idx]; }

    void emitEdges( ProblemSpecP edgesElement, int rank );

    DetailedTask* getNextInternalReadyTask();

    int numInternalReadyTasks();

    DetailedTask* getNextExternalReadyTask();

    int numExternalReadyTasks();

    void createScrubCounts();

    bool mustConsiderInternalDependencies() { return m_must_consider_internal_dependencies; }

    unsigned long getCurrentDependencyGeneration() { return m_current_dependency_generation; }

    const TaskGraph* getTaskGraph() const { return m_task_graph; }

    void setScrubCount( const Task::Dependency                     * req
                      ,        int                                   matl
                      ,  const Patch                               * patch
                      ,        std::vector<OnDemandDataWarehouseP> & dws
                      );

    int getExtraCommunication() { return m_extra_communication; }

    ParticleExchangeVar& getParticleSends() { return m_particle_sends; }

    ParticleExchangeVar& getParticleRecvs() { return m_particle_recvs; }
    
    void setTaskPriorityAlg( QueueAlg alg ) { m_task_priority_algorithm=alg; }

    QueueAlg getTaskPriorityAlg() { return m_task_priority_algorithm; }

#ifdef HAVE_CUDA
    void addFinalizeDevicePreparation(DetailedTask* dtask);
    void addInitiallyReadyDeviceTask( DetailedTask* dtask );
    void addCompletionPendingDeviceTask( DetailedTask* dtask );
    void addInitiallyReadyHostTask(DetailedTask* dtask);

    DetailedTask* getNextFinalizeDevicePreparationTask();
    DetailedTask* getNextInitiallyReadyDeviceTask();
    DetailedTask* getNextCompletionPendingDeviceTask();
    DetailedTask* getNextInitiallyReadyHostTask();

    DetailedTask* peekNextFinalizeDevicePreparationTask();
    DetailedTask* peekNextInitiallyReadyDeviceTask();
    DetailedTask* peekNextCompletionPendingDeviceTask();
    DetailedTask* peekNextInitiallyReadyHostTask();

    int numFinalizeDevicePreparation() { return finalizeDevicePreparationTasks_.size(); }
    int numInitiallyReadyDeviceTasks() { return initiallyReadyDeviceTasks_.size(); }
    int numCompletionPendingDeviceTasks() { return completionPendingDeviceTasks_.size(); }
    int numInitiallyReadyHostTasks() { return initiallyReadyHostTasks_.size(); }

    void createInternalDependencyBatch(DetailedTask* from,
                                   Task::Dependency* m_comp,
                                   const Patch* m_from_patch,
                                   DetailedTask* m_to_rank,
                                   Task::Dependency* m_req,
                                   const Patch *toPatch,
                                   int m_matl,
                                   const IntVector& m_low,
                                   const IntVector& m_high,
                                   DetailedDependency::CommCondition cond);
    // helper of possiblyCreateDependency
    DetailedDependency* findMatchingInternalDetailedDep(DependencyBatch* batch, DetailedTask* toTask, Task::Dependency* m_req,
                                         const Patch* m_from_patch, int m_matl, IntVector m_low, IntVector m_high,
                                         IntVector& totalLow, IntVector& totalHigh, DetailedDependency* &parent_dep);
#endif

  protected:

    friend class DetailedTask;

    void internalDependenciesSatisfied( DetailedTask* task );

    SchedulerCommon* getSchedulerCommon() { return m_sched; }

  private:

    // eliminate copy, assignment and move
    DetailedTasks( const DetailedTasks & )            = delete;
    DetailedTasks& operator=( const DetailedTasks & ) = delete;
    DetailedTasks( DetailedTasks && )                 = delete;
    DetailedTasks& operator=( DetailedTasks && )      = delete;

    void initializeBatches();

    void incrementDependencyGeneration();

    // helper of possiblyCreateDependency
    DetailedDependency* findMatchingDetailedDep(       DependencyBatch    * batch
                                               ,       DetailedTask       * toTask
                                               ,       Task::Dependency   * req
                                               , const Patch              * fromPatch
                                               ,       int                   matl
                                               ,       IntVector             low
                                               ,       IntVector             high
                                               ,       IntVector           & totalLow
                                               ,       IntVector           & totalHigh
                                               ,       DetailedDependency *& parent_dep
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

    std::map<int, int>   m_send_old_data_map;
    ParticleExchangeVar  m_particle_sends;
    ParticleExchangeVar  m_particle_recvs;


    SchedulerCommon               * m_sched;
    const ProcessorGroup          * m_proc_group;
    DetailedTasks                 * m_first_task;          // store the first so we can share the scrubCountTable
    std::vector<DetailedTask*>      m_tasks;
    KeyDatabase<Patch>              m_var_key_DB;
    KeyDatabase<Level>              m_level_key_DB;
    const TaskGraph               * m_task_graph;
    Task                          * m_send_old_data_task;
    std::vector<DetailedTask*>      m_local_tasks;
    std::vector<DependencyBatch*>   m_dep_batches;
    DetailedDependency            * m_init_requires;
    
// TODO - FIXME: Figure out why this was commented out long ago - APH 02/12/16
#if 0
    std::vector<DetailedReq*>  initreqs_;
#endif

    // True for mixed scheduler which needs to keep track of internal dependencies.
    bool m_must_consider_internal_dependencies;

    QueueAlg m_task_priority_algorithm;
    
    TaskQueue  m_ready_tasks{};
    TaskQueue  m_initially_ready_tasks{};
    TaskQueue  m_mpi_completed_tasks{};

    // This "generation" number is to keep track of which InternalDependency
    // links have been satisfied in the current timestep and avoids the
    // need to traverse all InternalDependency links to reset values.
    unsigned long m_current_dependency_generation;

    // for logging purposes - how much extra comm is going on
    int m_extra_communication;

    ScrubCountTable m_scrub_table_count;


#ifdef HAVE_CUDA
    TaskPQueue            initiallyReadyDeviceTasks_;       // initially ready, h2d copies pending
    TaskPQueue            finalizeDevicePreparationTasks_;  // h2d copies completed, need to mark gpu data as valid and copy gpu ghost cell data internall on device
    TaskPQueue            completionPendingDeviceTasks_;    // execution and d2h copies pending
    TaskPQueue            initiallyReadyHostTasks_;         // initially ready cpu m_task, d2h copies pending

    mutable CrowdMonitor  deviceFinalizePreparationQueueLock_;
    mutable CrowdMonitor  deviceReadyQueueLock_;
    mutable CrowdMonitor  deviceCompletedQueueLock_;
    mutable CrowdMonitor  hostReadyQueueLock_;
#endif


    friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );
    friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDependency& task );

  }; // class DetailedTasks


  std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );
  std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDependency& task );


} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

