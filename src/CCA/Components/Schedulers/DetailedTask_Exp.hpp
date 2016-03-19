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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_EXP_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_EXP_H

#include <CCA/Components/Schedulers/DependencyBatch_Exp.hpp>
#include <CCA/Components/Schedulers/DetailedDependency_Exp.hpp>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
#include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#include <set>
#endif

#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace Uintah {

class ProcessorGroup;
class TaskGraph;
class DetailedTasks;

#ifdef HAVE_CUDA
struct TaskGpuDataWarehouses{
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


  // specifies the type of task this is:
  //   * normal executes on either the patches cells or the patches coarse cells
  //   * fine executes on the patches fine cells (for example coarsening)
  enum ProfileType {
     Normal
   , Fine
  };

  void setProfileType( ProfileType type ) { m_profile_type=type; }

  ProfileType getProfileType() { return m_profile_type; }

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

  const Task* getTask() const { return m_task; }

  const PatchSubset* getPatches() const { return m_patches; }

  const MaterialSubset* getMaterials() const { return m_matls; }

  void assignResource(int idx) { m_resource_index = idx; }

  int getAssignedResourceIndex() const { return m_resource_index; }

  void assignStaticOrder( int i ) { m_static_order = i; }

  int getStaticOrder() const { return m_static_order; }

  DetailedTasks* getTaskGroup() const { return m_task_group; }

  std::map<DependencyBatch*, DependencyBatch*>& getRequires() { return m_requires; }

  std::map<DependencyBatch*, DependencyBatch*>& getInternalRequires() { return m_internal_requires; }

  DependencyBatch* getComputes() const { return m_comp_head; }

  DependencyBatch* getInternalComputes() const { return m_internal_comp_head; }

  void findRequiringTasks( const VarLabel* var, std::list<DetailedTask*>& requiringTasks );

  void emitEdges( ProblemSpecP edgesElement );

  bool addInternalRequires( DependencyBatch*);

  void addInternalComputes( DependencyBatch* );

  bool addRequires( DependencyBatch* );

  void addComputes( DependencyBatch* );

  void addInternalDependency( DetailedTask* prerequisiteTask, const VarLabel* var );

  void resetDependencyCounts();

  bool isInitiated() const { return m_initiated; }

  void markInitiated()
  {
    m_wait_timer.start();
    m_initiated = true;
  }

  void incrementExternalDepCount() { m_external_dependency_count++; }

  void decrementExternalDepCount() { m_external_dependency_count--; }

  // external dependencies will count how many messages this task is waiting for.
  // When it hits 0, it is ready to run
  void checkExternalDepCount();

  int getExternalDepCount() { return m_external_dependency_count; }

  bool areInternalDependenciesSatisfied() { return (m_num_pending_internal_dependencies == 0); }

  double task_wait_time() const { return m_wait_timer().seconds(); }

  double task_exec_time() const { return m_exec_timer().seconds(); }


#ifdef HAVE_CUDA

  void assignDevice (unsigned int device);

  //unsigned int getDeviceNum() const;

  //Most tasks will only run on one device.
  //But some, such as the data archiver task or send old data could run on multiple devices.
  //This is not a good idea.  A task should only run on one device.  But the capability for a task
  //to run on multiple nodes exists.
  std::set<unsigned int> getDeviceNums() const;
  std::map<unsigned int, TaskGpuDataWarehouses> TaskGpuDWs;


  //bool queryCudaStreamCompletionForThisTask();

  //void setCudaStreamForThisTask(cudaStream_t* s);

  void setCudaStreamForThisTask(unsigned int deviceNum, cudaStream_t* s);

  void clearCudaStreamsForThisTask();

  //bool checkCudaStreamDoneForThisTask() const;

  bool checkCudaStreamDoneForThisTask(unsigned int deviceNum) const;

  bool checkAllCudaStreamsDoneForThisTask() const;

  void setTaskGpuDataWarehouse(unsigned int deviceNum, Task::WhichDW DW, GPUDataWarehouse* TaskDW);

  GPUDataWarehouse* getTaskGpuDataWarehouse(unsigned int deviceNum, Task::WhichDW DW);

  void deleteTaskGpuDataWarehouses();

  cudaStream_t* getCudaStreamForThisTask(unsigned int deviceNum) const;

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

  // Called when prerequisite tasks (dependencies) call done.
  void dependencySatisfied( InternalDependency* dep );

  // eliminate copy, assignment and move
  DetailedTask( const DetailedTask & )            = delete;
  DetailedTask& operator=( const DetailedTask & ) = delete;
  DetailedTask( DetailedTask && )                 = delete;
  DetailedTask& operator=( DetailedTask && )      = delete;

  Task                                         * m_task{};
  const PatchSubset                            * m_patches{};
  const MaterialSubset                         * m_matls{};
  std::map<DependencyBatch*, DependencyBatch*>   m_requires{};
  std::map<DependencyBatch*, DependencyBatch*>   m_internal_requires{};
  DependencyBatch                              * m_comp_head{};
  DependencyBatch                              * m_internal_comp_head{};
  DetailedTasks                                * m_task_group{};

  bool m_initiated{};
  bool m_externally_ready{};
  int  m_external_dependency_count{};

  mutable std::string m_name; // doesn't get set until getName() is called the first time.

  // Internal dependencies are dependencies within the same process.
  std::list<InternalDependency> m_internal_dependencies{};

  // internalDependents will point to InternalDependency's in the
  // m_internal_dependencies list of the requiring DetailedTasks.
  std::map<DetailedTask*, InternalDependency*> m_internal_dependents{};

  unsigned long   m_num_pending_internal_dependencies{};
  std::mutex      m_internal_dependency_lock{};

  int m_resource_index;
  int m_static_order;

  RuntimeStats::TaskExecTimer m_exec_timer{this};
  RuntimeStats::TaskWaitTimer m_wait_timer{this};

  bool operator<( const DetailedTask& other );

  ProfileType m_profile_type{};

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

std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_EXP_H
