/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_H

#include <CCA/Components/Schedulers/DetailedDependency.h>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <Core/Grid/Task.h>

#include <sci_defs/gpu_defs.h>

#if defined(UINTAH_USING_GPU)
  #include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
  #include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
  #include <CCA/Components/Schedulers/GPUMemoryPool.h>
#ifdef USE_KOKKOS_INSTANCE
#elif defined(HAVE_CUDA) // CUDA only when using streams
  #include <CCA/Components/Schedulers/GPUStreamPool.h>
#endif
#endif

#include <atomic>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <vector>


namespace Uintah {

class ProcessorGroup;
class DependencyBatch;
class DetailedTasks;


//_____________________________________________________________________________
//
#if defined(UINTAH_USING_GPU)
  struct TaskGpuDataWarehouses {
    GPUDataWarehouse* TaskGpuDW[2];
  };
#endif


//_____________________________________________________________________________
//
enum ProfileType {
    Normal
  , Fine
};


//_____________________________________________________________________________
//
struct InternalDependency {

  InternalDependency(       DetailedTask * prerequisiteTask
                    ,       DetailedTask * dependentTask
                    , const VarLabel     * var
                    ,       long           satisfiedGeneration
                    )
    : m_prerequisite_task(prerequisiteTask)
    , m_dependent_task(dependentTask)
    , m_satisfied_generation(satisfiedGeneration)
  {
    addVarLabel(var);
  }

  void addVarLabel(const VarLabel* var)
  {
    m_vars.insert(var);
  }

  DetailedTask                                 * m_prerequisite_task;
  DetailedTask                                 * m_dependent_task;
  std::set<const VarLabel*, VarLabel::Compare>   m_vars;
  unsigned long                                  m_satisfied_generation;

};


//_____________________________________________________________________________
//
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
    m_profile_type = type;
  }

  ProfileType getProfileType()
  {
    return m_profile_type;
  }

  struct labelPatchMatlLevelDw {
    std::string label;
    int         patchID;
    int         matlIndx;
    int         levelIndx;
    int         dwIndex;

    labelPatchMatlLevelDw(const char * label, int patchID, int matlIndx, int levelIndx, int dwIndex) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
      this->levelIndx = levelIndx;
      this->dwIndex = dwIndex;
    }

    //This so it can be used in an STL map
    bool operator<(const labelPatchMatlLevelDw& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx < right.matlIndx)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx) && (this->levelIndx < right.levelIndx)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx) && (this->levelIndx == right.levelIndx) && this->dwIndex < right.dwIndex) {
        return true;
      } else {
        return false;
      }
    }
  };

#if defined(UINTAH_USING_GPU)
  struct delayedCopyingInfo {
    delayedCopyingInfo( GpuUtilities::LabelPatchMatlLevelDw   lpmld_
                      , DeviceGridVariableInfo                devGridVarInfo_
                      , void                                * device_ptr_
                      , void                                * host_ptr_
                      , size_t                                size_
                      )
      : lpmld(lpmld_)
      , devGridVarInfo(devGridVarInfo_)
      , device_ptr(device_ptr_)
      , host_ptr(host_ptr_)
      , size(size_)
    {}

    GpuUtilities::LabelPatchMatlLevelDw lpmld;
    DeviceGridVariableInfo              devGridVarInfo;
    void *                              device_ptr;
    void *                              host_ptr;
    size_t                              size;
  };
#endif

  void doit( const ProcessorGroup                      * pg
           ,       std::vector<OnDemandDataWarehouseP> & oddws
           ,       std::vector<DataWarehouseP>         & dws
           ,       CallBackEvent                         event = CallBackEvent::CPU
           );

  // Called after doit and MPI data sent (packed in buffers) finishes.
  // Handles internal dependencies and scrubbing. Called after doit finishes.
  void done( std::vector<OnDemandDataWarehouseP> & dws );

  std::string getName() const;

  const Task*           getTask() const {      return m_task; }
  const PatchSubset*    getPatches() const {   return m_patches; }
  const MaterialSubset* getMaterials() const { return m_matls; }

  void assignResource( int idx ) { m_resource_index = idx; }
  int  getAssignedResourceIndex() const { return m_resource_index; }

  void assignStaticOrder( int i )  { m_static_order = i; }
  int  getStaticOrder() const { return m_static_order; }

  DetailedTasks* getTaskGroup() const { return m_task_group; }

  std::map<DependencyBatch*, DependencyBatch*>& getRequires() { return m_reqs; }
  std::map<DependencyBatch*, DependencyBatch*>& getInternalRequires() { return m_internal_reqs; }

  DependencyBatch* getComputes() const { return m_comp_head; }
  DependencyBatch* getInternalComputes() const { return m_internal_comp_head; }

  void findRequiringTasks( const VarLabel * var , std::list<DetailedTask*> & requiringTasks );

  void emitEdges( ProblemSpecP edgesElement );

  bool addInternalRequires( DependencyBatch * req );

  void addInternalComputes( DependencyBatch * comp );

  bool addRequires( DependencyBatch * req );

  void addComputes( DependencyBatch * comp );

  void addInternalDependency( DetailedTask * prerequisiteTask, const VarLabel * var );

  // External dependencies will count how many messages this task is
  // waiting for.  When it hits 0, we can add it to the
  // DetailedTasks::mpiCompletedTasks list.
  void resetDependencyCounts();

  void markInitiated()
  {
    m_wait_timer.start();
    m_initiated.store( true, std::memory_order_seq_cst );
  }

  void incrementExternalDepCount() { m_external_dependency_count.fetch_add( 1, std::memory_order_seq_cst ); }
  void decrementExternalDepCount() { m_external_dependency_count.fetch_sub( 1, std::memory_order_seq_cst ); }

  void checkExternalDepCount();
  int  getExternalDepCount() { return m_external_dependency_count.load(std::memory_order_seq_cst); }

  bool areInternalDependenciesSatisfied() { return ( m_num_pending_internal_dependencies == 0 ); }

  double task_wait_time() const { return m_wait_timer().seconds(); }
  double task_exec_time() const { return m_exec_timer().seconds(); }

//-----------------------------------------------------------------------------
#if defined(UINTAH_USING_GPU)

  typedef std::set<unsigned int>       deviceNumSet;
  typedef deviceNumSet::const_iterator deviceNumSetIter;

#ifdef TASK_MANAGES_EXECSPACE
  // These methods are defined so that the UnifiedScheduler
  // compiles. However the UnifiedScheduler is not used.
  void assignDevice( unsigned int device )
  {
    printf("ERROR: DetailedTask::assignDevice - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::assignDevice - Should not be called.", __FILE__, __LINE__));
  };

  deviceNumSet getDeviceNums() const
  {
    printf("ERROR: DetailedTask::getDeviceNums - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::getDeviceNums - Should not be called.", __FILE__, __LINE__));
  };

#ifdef USE_KOKKOS_INSTANCE

  // These three methods are pass through methods to the actual task
  // similar to doit.
  void clearKokkosInstancesForThisTask();

  bool checkAllKokkosInstancesDoneForThisTask() const;

  // These methods are defined so that the UnifiedScheduler
  // compiles. However the UnifiedScheduler is not used.
  void setCudaStreamForThisTask( unsigned int deviceNum, cudaStream_t * s )
  {
    printf("ERROR: DetailedTask::setCudaStreamForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::setCudaStreamForThisTask - Should not be called.", __FILE__, __LINE__));
  };

  cudaStream_t* getCudaStreamForThisTask( unsigned int deviceNum ) const
  {
    printf("ERROR: DetailedTask::getCudaStreamForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::getCudaStreamForThisTask - Should not be called.", __FILE__, __LINE__));
  };

  void reclaimCudaStreamsIntoPool()
  {
    printf("ERROR: DetailedTask::reclaimCudaStreamsIntoPool - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::reclaimCudaStreamsIntoPool - Should not be called.", __FILE__, __LINE__));
  };

  void clearCudaStreamsForThisTask()
  {
    printf("ERROR: DetailedTask::clearCudaStreamsForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::clearCudaStreamsForThisTask - Should not be called.", __FILE__, __LINE__));
  };

  bool checkAllCudaStreamsDoneForThisTask()
  {
    printf("ERROR: DetailedTask::checkAllCudaStreamsDoneForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::checkAllCudaStreamsDoneForThisTask - Should not be called.", __FILE__, __LINE__));
  };
#else

  // Pass through methods
  void reclaimCudaStreamsIntoPool();

  void clearCudaStreamsForThisTask();

  bool checkCudaStreamDoneForThisTask( unsigned int deviceNum ) const;

  bool checkAllCudaStreamsDoneForThisTask() const;

  void setCudaStreamForThisTask( unsigned int deviceNum, cudaStream_t * s )
  {
    printf("ERROR: DetailedTask::setCudaStreamForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::setCudaStreamForThisTask - Should not be called.", __FILE__, __LINE__));
  };

  cudaStream_t* getCudaStreamForThisTask( unsigned int deviceNum ) const
  {
    printf("ERROR: DetailedTask::getCudaStreamForThisTask - Should not be called.\n");
    SCI_THROW(InternalError("DetailedTask::getCudaStreamForThisTask - Should not be called.", __FILE__, __LINE__));
  };
#endif
#else
  typedef std::map<unsigned int, cudaStream_t*> cudaStreamMap;
  typedef cudaStreamMap::const_iterator         cudaStreamMapIter;

  void assignDevice( unsigned int device );

  // Most tasks will only run on one device.

  // But some, such as the data archiver task or send_old_data could
  // run on multiple devices.

  // This is not a good idea.  A task should only run on one device.
  // But the capability for a task to run on multiple nodes exists.
  deviceNumSet getDeviceNums() const;

  void setCudaStreamForThisTask( unsigned int deviceNum, cudaStream_t * s );

  cudaStream_t* getCudaStreamForThisTask( unsigned int deviceNum ) const;

  void reclaimCudaStreamsIntoPool();

  void clearCudaStreamsForThisTask();

  bool checkCudaStreamDoneForThisTask( unsigned int deviceNum ) const;

  bool checkAllCudaStreamsDoneForThisTask() const;
#endif

  // Task GPU date warehouses
  std::map<unsigned int, TaskGpuDataWarehouses> TaskGpuDWs;

  void setTaskGpuDataWarehouse( unsigned int       deviceNum
                              , Task::WhichDW      DW
                              , GPUDataWarehouse * TaskDW
                              );

  GPUDataWarehouse* getTaskGpuDataWarehouse( unsigned int deviceNum, Task::WhichDW DW );

  void deleteTaskGpuDataWarehouses();

  DeviceGridVariables& getDeviceVars() { return deviceVars; }

  DeviceGridVariables& getTaskVars() { return taskVars; }

  DeviceGhostCells&    getGhostVars() { return ghostVars; }

  DeviceGridVariables& getVarsToBeGhostReady() { return varsToBeGhostReady; }

  DeviceGridVariables& getVarsBeingCopiedByTask() { return varsBeingCopiedByTask; }

  inline std::vector<labelPatchMatlLevelDw>& getVarsNeededOnHost() { return varsNeededOnHost;}
  inline std::vector<delayedCopyingInfo>& getDelayedCopyingVars() { return delayedCopyingVars;}
  inline int getDelayedCopy(){ return delayedCopy; }
  inline void setDelayedCopy(int val){ delayedCopy = val; }

  void clearPreparationCollections();

  void addTempHostMemoryToBeFreedOnCompletion( void * ptr );

  void addTempCudaMemoryToBeFreedOnCompletion( unsigned int device_ptr, void * ptr );

  void deleteTemporaryTaskVars();

#endif
//-----------------------------------------------------------------------------

  static std::string myRankThread();

protected:

  friend class TaskGraph;


private:

  // Eliminate copy, assignment and move
  DetailedTask(const DetailedTask &)            = delete;
  DetailedTask& operator=(const DetailedTask &) = delete;
  DetailedTask(DetailedTask &&)                 = delete;
  DetailedTask& operator=(DetailedTask &&)      = delete;

  // Called by done()
  void scrub( std::vector<OnDemandDataWarehouseP> & dws );

  // Called when prerequisite tasks (dependencies) call done.
  void dependencySatisfied(InternalDependency * dep);

  Task                                         * m_task { nullptr };
  const PatchSubset                            * m_patches { nullptr };
  const MaterialSubset                         * m_matls { nullptr };
  std::map<DependencyBatch*, DependencyBatch*>   m_reqs;
  std::map<DependencyBatch*, DependencyBatch*>   m_internal_reqs;
  DependencyBatch                              * m_comp_head { nullptr };
  DependencyBatch                              * m_internal_comp_head { nullptr };
  DetailedTasks                                * m_task_group { nullptr };

  std::atomic<bool> m_initiated { false };
  std::atomic<bool> m_externally_ready { false };
  std::atomic<int>  m_external_dependency_count { 0 };

  mutable std::string m_name;  // Doesn't get set until getName() is
                               // called the first time.

  // Internal dependencies are dependencies within the same process.
  std::list<InternalDependency> m_internal_dependencies;

  // Internaldependents will point to InternalDependency's in the
  // internalDependencies list of the requiring DetailedTasks.
  std::map<DetailedTask*, InternalDependency*> m_internal_dependents;

  unsigned long m_num_pending_internal_dependencies { 0 };

  int m_resource_index { -1 };
  int m_static_order   { -1 };

  // Specifies the type of task this is:
  //   * Normal executes on either the patches cells or the patches coarse cells
  //   * Fine   executes on the patches fine cells (for example coarsening)
  ProfileType m_profile_type { Normal };

  RuntimeStats::TaskExecTimer m_exec_timer{this};
  RuntimeStats::TaskWaitTimer m_wait_timer{this};

  bool operator<(const DetailedTask & other);


//-----------------------------------------------------------------------------
#if defined(UINTAH_USING_GPU)
private:
//  bool         m_deviceExternallyReady{false};
//  bool         m_completed{false};
//  unsigned int m_deviceNum{0};

#ifdef TASK_MANAGES_EXECSPACE
  // Defined in Task.h
#else
  deviceNumSet  m_deviceNums;
  cudaStreamMap m_cudaStreams;
#endif
  // Store information about each set of grid variables.  This will
  // help later when we figure out the best way to store data into the
  // GPU.  It may be stored contiguously.  It may handle material
  // data.  It just helps to gather it all up into a collection prior
  // to copying data.
  DeviceGridVariables deviceVars;  // Holds variables that will need
                                   // to be copied into the GPU
  DeviceGridVariables taskVars;    // Holds variables that will be
                                   // needed for a GPU task (a Task DW
                                   // has a snapshot of all important
                                   // pointer info from the host-side
                                   // GPU DW)
  DeviceGhostCells ghostVars;      // Holds ghost cell meta data copy
                                   // information

  DeviceGridVariables varsToBeGhostReady;  // Holds a list of vars
                                           // this task is managing to
                                           // ensure their ghost cells
                                           // will be ready.  This
                                           // means this task is the
                                           // exclusive ghost cell
                                           // gatherer and ghost cell
                                           // validator for any
                                           // label/patch/matl/level
                                           // vars it has listed in
                                           // here But it is NOT the
                                           // exclusive copier.
                                           // Because some ghost cells
                                           // from one patch may be
                                           // used by two or more
                                           // destination patches.  We
                                           // only want to copy ghost
                                           // cells once.

  DeviceGridVariables varsBeingCopiedByTask;  // Holds a list of the
                                              // vars that this task
                                              // is actually copying
                                              // into the GPU.

  struct gpuMemoryPoolDevicePtrItem {

    unsigned int   device_id;
    void         * ptr;

    gpuMemoryPoolDevicePtrItem( unsigned int device_id, void * ptr )
    {
      this->device_id = device_id;
      this->ptr       = ptr;
    }

    // This so it can be used in an STL map
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

  std::vector<labelPatchMatlLevelDw> varsNeededOnHost;
  std::vector<delayedCopyingInfo> delayedCopyingVars;
  int delayedCopy{0};

//-----------------------------------------------------------------------------

public:
    // int m_num_partitions{0};
    // int m_threads_per_partition{0};

    using DeviceVarDest = GpuUtilities::DeviceVarDestination;

    void assignStatusFlagsToPrepareACpuTask();

    void assignDevicesAndStreams();

    void assignDevicesAndStreamsFromGhostVars();

    void findIntAndExtGpuDependencies( std::vector<OnDemandDataWarehouseP> & m_dws
                                     , std::set<std::string> &m_no_copy_data_vars
                                     , const VarLabel * m_reloc_new_pos_label
                                     , const VarLabel * m_parent_reloc_new_pos_label
                                     , int iteration
                                     , int t_id );

    void prepareGpuDependencies(       DependencyBatch       * batch
                               , const VarLabel              * pos_var
                               ,       OnDemandDataWarehouse * dw
                               ,       OnDemandDataWarehouse * old_dw
                               , const DetailedDep           * dep
                               ,       DeviceVarDest           des
                               );

    void createTaskGpuDWs();
    void   syncTaskGpuDWs();

    // void gpuInitialize( bool reset = false );

    Uintah::MasterLock * varLock {nullptr};

    void performInternalGhostCellCopies();

    void copyAllGpuToGpuDependences(std::vector<OnDemandDataWarehouseP> & m_dws);

    void copyAllExtGpuDependenciesToHost(std::vector<OnDemandDataWarehouseP> & m_dws);

    void initiateH2DCopies(std::vector<OnDemandDataWarehouseP> & m_dws);

    void turnIntoASuperPatch(GPUDataWarehouse* const       gpudw,
                             const Level* const            level,
                             const IntVector&              low,
                             const IntVector&              high,
                             const VarLabel* const         label,
                             const Patch * const           patch,
                             const int                     matlIndx,
                             const int                     levelID );

    void prepareDeviceVars(std::vector<OnDemandDataWarehouseP> & m_dws);

    void copyDelayedDeviceVars();

    // Check if the main patch is valid, not ghost cells.
    bool delayedDeviceVarsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void prepareTaskVarsIntoTaskDW(std::vector<OnDemandDataWarehouseP> & m_dws);

    void prepareGhostCellsIntoTaskDW();

    void markDeviceRequiresAndModifiesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markHostAsInvalid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markDeviceGhostsAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markHostComputesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markDeviceComputesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markDeviceModifiesGhostAsInvalid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markHostRequiresAndModifiesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void markDeviceAsInvalidHostAsValid(std::vector<OnDemandDataWarehouseP> & m_dws);

    void initiateD2HForHugeGhostCells(std::vector<OnDemandDataWarehouseP> & m_dws);

    void initiateD2H(const ProcessorGroup                * d_myworld,
                           std::vector<OnDemandDataWarehouseP> & m_dws);

    bool ghostCellsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws);

    bool allHostVarsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws);

    bool allGPUVarsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws);

    // struct GPUGridVariableInfo {

    //   GPUGridVariableInfo( DetailedTask * dtask
    //                      , double       * ptr
    //                      , IntVector      size
    //                      , int            device
    //                      )
    //     : m_dtask{dtask}
    //     , m_ptr{ptr}
    //     , m_size{size}
    //     , m_device{device}
    //   {}

    //   DetailedTask * m_dtask;
    //   double       * m_ptr;
    //   IntVector      m_size;
    //   int            m_device;
    // };

    // std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_device_requires_ptrs;
    // std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_device_computes_ptrs;
    // std::map<std::string, GPUGridVariableInfo>          m_device_computes_temp_ptrs;
    // std::vector<VarLabel*>                              m_tmp_var_labels;

    // std::vector<GPUGridVariableInfo>                    m_device_requires_allocation_ptrs;
    // std::vector<GPUGridVariableInfo>                    m_device_computes_allocation_ptrs;
    // std::vector<double*>                                m_host_computes_allocation_ptrs;

    // std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_host_requires_ptrs;
    // std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_host_computes_ptrs;
    // std::vector<std::queue<cudaEvent_t*> >              m_idle_events;

    // int  m_num_devices;
    // int  m_current_device;

    // ARS - 18/11/22 - Not used and specific to ICE??
    // std::vector< std::string > m_material_names;

    struct labelPatchMatlDependency {

        labelPatchMatlDependency( const char          * label
                                ,       int             patchID
                                ,       int             matlIndex
                                ,       Task::DepType   depType
                                )
          : m_label{label}
          , m_patchID{patchID}
          , m_matlIndex{matlIndex}
          , m_depType{depType}
        {}

        // this so it can be used in an STL map
        bool operator<(const labelPatchMatlDependency& right) const
        {
          if (m_label < right.m_label) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID < right.m_patchID)) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndex < right.m_matlIndex)) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndex == right.m_matlIndex) && (m_depType < right.m_depType)) {
            return true;
          }
          else {
            return false;
          }
        }

        std::string   m_label;
        int           m_patchID;
        int           m_matlIndex;
        Task::DepType m_depType;
    };

#endif

}; // class DetailedTask

std::ostream& operator<<( std::ostream & out, const Uintah::DetailedTask & task );

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_H
