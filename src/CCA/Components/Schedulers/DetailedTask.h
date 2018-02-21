/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
#endif

#include <Core/Grid/Task.h>

#include <sci_defs/cuda_defs.h>

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
#ifdef HAVE_CUDA

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

  void doit( const ProcessorGroup                      * pg
           ,       std::vector<OnDemandDataWarehouseP> & oddws
           ,       std::vector<DataWarehouseP>         & dws
           ,       Task::CallBackEvent                   event = Task::CPU
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

  // external dependencies will count how many messages this task is waiting for.
  // When it hits 0, we can add it to the  DetailedTasks::mpiCompletedTasks list.
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

  cudaStream_t*        getCudaStreamForThisTask( unsigned int deviceNum ) const;

  DeviceGridVariables& getDeviceVars() { return deviceVars; }

  DeviceGridVariables& getTaskVars() { return taskVars; }

  DeviceGhostCells&    getGhostVars() { return ghostVars; }

  DeviceGridVariables& getVarsToBeGhostReady() { return varsToBeGhostReady; }

  DeviceGridVariables& getVarsBeingCopiedByTask() { return varsBeingCopiedByTask; }

  void clearPreparationCollections();

  void addTempHostMemoryToBeFreedOnCompletion( void * ptr );

  void addTempCudaMemoryToBeFreedOnCompletion( unsigned int device_ptr, void * ptr );

  void deleteTemporaryTaskVars();

#endif
//-----------------------------------------------------------------------------


protected:

  friend class TaskGraph;


private:

  // eliminate copy, assignment and move
  DetailedTask(const DetailedTask &)            = delete;
  DetailedTask& operator=(const DetailedTask &) = delete;
  DetailedTask(DetailedTask &&)                 = delete;
  DetailedTask& operator=(DetailedTask &&)      = delete;

  // called by done()
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

  mutable std::string m_name;  // doesn't get set until getName() is called the first time.

  // Internal dependencies are dependencies within the same process.
  std::list<InternalDependency> m_internal_dependencies;

  // internalDependents will point to InternalDependency's in the
  // internalDependencies list of the requiring DetailedTasks.
  std::map<DetailedTask*, InternalDependency*> m_internal_dependents;

  unsigned long m_num_pending_internal_dependencies { 0 };

  int m_resource_index { -1 };
  int m_static_order   { -1 };

  // specifies the type of task this is:
  //   * Normal executes on either the patches cells or the patches coarse cells
  //   * Fine   executes on the patches fine cells (for example coarsening)
  ProfileType m_profile_type { Normal };

  RuntimeStats::TaskExecTimer m_exec_timer{this};
  RuntimeStats::TaskWaitTimer m_wait_timer{this};

  bool operator<(const DetailedTask & other);


//-----------------------------------------------------------------------------
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
                                           // This means this task is the exclusive ghost cell gatherer and ghost cell validator for any
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

#endif
//-----------------------------------------------------------------------------


}; // class DetailedTask

std::ostream& operator<<( std::ostream & out, const Uintah::DetailedTask & task );

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_H

