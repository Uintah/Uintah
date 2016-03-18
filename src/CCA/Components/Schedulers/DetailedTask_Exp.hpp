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

#include <CCA/Components/Schedulers/DetailedDep_Exp.hpp>
#include <CCA/Components/Schedulers/DependencyBatch_Exp.hpp>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>

#include <Core/Grid/Task.h>
#include <Core/Grid/Patch.h>

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
class TaskGraph;
class DetailedTasks;

#ifdef HAVE_CUDA
struct TaskGpuDataWarehouses{
public:
  GPUDataWarehouse* TaskGpuDW[2];
};

#endif


enum ProfileType {
   Normal
 , Fine
};


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

  void markInitiated()
  {
    m_wait_timer.start();
    initiated_ = true;
  }

  void incrementExternalDepCount() { externalDependencyCount_++; }

  void decrementExternalDepCount() { externalDependencyCount_--; }

  void checkExternalDepCount();

  int getExternalDepCount() { return externalDependencyCount_; }

  bool areInternalDependenciesSatisfied() { return (numPendingInternalDependencies == 0); }

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

  RuntimeStats::TaskExecTimer m_exec_timer{this};
  RuntimeStats::TaskWaitTimer m_wait_timer{this};

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

std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );

} // namespace Uintah

#endif  // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASK_EXP_H
