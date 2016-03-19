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

#ifdef UINTAH_USING_EXPERIMENTAL

#include <CCA/Components/Schedulers/DetailedTask_Exp.cpp>

#else

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/CommRecMPI.h>
#include <CCA/Components/Schedulers/MemoryLog.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/config_defs.h>
#include <sci_defs/cuda_defs.h>

extern Uintah::Mutex coutLock;

namespace Uintah {

namespace {

Dout mpidbg(             "MPIDBG",    false );
DebugStream dbg(         "DetailedTasks", false);
DebugStream scrubout(    "Scrubbing",     false);
DebugStream messagedbg(  "MessageTags",   false);
DebugStream internaldbg( "InternalDeps",  false);
DebugStream dwdbg(       "DetailedDWDBG", false);
DebugStream waitout(     "WaitTimes",     false);

Dout g_detailed_dbg{ "DetailedTasksDBG", false };

std::string dbgScrubVar   = ""; // for debugging - set the var name to watch one in the scrubout
int         dbgScrubPatch = -1;

} // namespace


//_____________________________________________________________________________
//
DetailedTask::DetailedTask(       Task*           task,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DetailedTasks*  taskGroup )
  :   m_task(task),
      m_patches(patches),
      m_matls(matls),
      m_comp_head(0),
      m_internal_comp_head(0),
      m_task_group(taskGroup),
      m_num_pending_internal_dependencies(0),
      m_resource_index(-1),
      m_static_order(-1),
      m_profile_type(Normal)
{
  if (patches) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(patches->getVector().begin(), patches->getVector().end(), Patch::Compare()) );
    patches->addReference();
  }
  if (matls) {
    // patches and matls must be sorted
    ASSERT( std::is_sorted(matls->getVector().begin(), matls->getVector().end()) );
    matls->addReference();
  }
#ifdef HAVE_CUDA
  deviceExternallyReady_ = false;
  completed_             = false;
  deviceNum_             = -1;
#endif
}

//_____________________________________________________________________________
//
DetailedTask::~DetailedTask()
{
  if (m_patches && m_patches->removeReference()) {
    delete m_patches;
  }
  if (m_matls && m_matls->removeReference()) {
    delete m_matls;
  }
}

//_____________________________________________________________________________
//
void DetailedTask::doit( const ProcessorGroup*                 pg,
                    std::vector<OnDemandDataWarehouseP>& oddws,
                    std::vector<DataWarehouseP>&         dws,
                          Task::CallBackEvent             event /* = Task::CPU */ )
{
  m_wait_timer.stop();
  m_exec_timer.start();

  for (int i = 0; i < (int)dws.size(); i++) {
    if (oddws[i] != 0) {
      oddws[i]->pushRunningTask(m_task, &oddws);
    }
  }

#ifdef HAVE_CUDA
  // determine if task will be executed on CPU or device, e.g. GPU or MIC
  if (m_task->usesDevice()) {
    //Run the GPU task.  Technically the engine has structure to run one task on multiple devices if
    //that task had patches on multiple devices.  So run the task once per device.  As often as possible,
    //we want to design tasks so each task runs on only once device, instead of a one to many relationship.
    for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums_.begin(); deviceNums_it != deviceNums_.end(); ++deviceNums_it) {
      const unsigned int currentDevice = *deviceNums_it;
      OnDemandDataWarehouse::uintahSetCudaDevice(currentDevice);
      m_task->doit(event, pg, m_patches, m_matls, dws,
                 getTaskGpuDataWarehouse(currentDevice, Task::OldDW),
                 getTaskGpuDataWarehouse(currentDevice, Task::NewDW),
                 getCudaStreamForThisTask(currentDevice), currentDevice);
    }
  }
  else {
    m_task->doit(event, pg, m_patches, m_matls, dws, NULL, NULL, NULL, -1);
  }
#else
  m_task->doit(event, pg, m_patches, m_matls, dws, NULL, NULL, NULL, -1);
#endif

  for (int i = 0; i < (int)dws.size(); i++) {
    if (oddws[i] != 0) {
      oddws[i]->checkTasksAccesses(m_patches, m_matls);
      oddws[i]->popRunningTask();
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::scrub( std::vector<OnDemandDataWarehouseP>& dws )
{
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " Starting scrub after task: " << *this << '\n';
  }
  const Task* task = getTask();

  const std::set<const VarLabel*, VarLabel::Compare>& initialRequires = m_task_group->getSchedulerCommon()->getInitialRequiredVars();
  const std::set<std::string>& unscrubbables = m_task_group->getSchedulerCommon()->getNoScrubVars();

  // Decrement the scrub count for each of the required variables
  for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->next) {
    TypeDescription::Type type = req->var->typeDescription()->getType();
    Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
    if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
      int dw = req->mapDataWarehouse();

      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if (scrubmode == DataWarehouse::ScrubComplete
          || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(req->var) == initialRequires.end())) {

        if (unscrubbables.find(req->var->getName()) != unscrubbables.end())
          continue;

        constHandle<PatchSubset> patches = req->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(getMaterials());
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          Patch::selectType neighbors;
          IntVector low, high;

          if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel || req->numGhostCells == 0) {
            // we already have the right patches
            neighbors.push_back(patch);
          }
          else {
            patch->computeVariableExtents(type, req->var->getBoundaryLayer(), req->gtype, req->numGhostCells, neighbors, low, high);
          }

          for (int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];

            if (patch->getLevel()->getIndex() > 0 && patch != neighbor && req->patches_dom == Task::ThisLevel) {
              // don't scrub on AMR overlapping patches...
              IntVector l = low, h = high;
              l = Max(neighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), low);
              h = Min(neighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), high);
              patch->cullIntersection(basis, req->var->getBoundaryLayer(), neighbor->getRealPatch(), l, h);
              if (l == h)
                continue;
            }
            if (req->patches_dom == Task::FineLevel) {
              // don't count if it only overlaps extra cells
              IntVector l = patch->getExtraLowIndex(basis, IntVector(0, 0, 0)), h = patch->getExtraHighIndex(basis,
                                                                                                             IntVector(0, 0, 0));
              IntVector fl = neighbor->getLowIndex(basis), fh = neighbor->getHighIndex(basis);
              IntVector il = Max(l, neighbor->getLevel()->mapCellToCoarser(fl));
              IntVector ih = Min(h, neighbor->getLevel()->mapCellToCoarser(fh));
              if (ih.x() <= il.x() || ih.y() <= il.y() || ih.z() <= il.z()) {
                continue;
              }
            }

            for (int m = 0; m < matls->size(); m++) {
              int count;
              try {
                // there are a few rare cases in an AMR framework where you require from an OldDW, but only
                // ones internal to the W-cycle (and not the previous timestep) which can have variables not exist in the OldDW.
                count = dws[dw]->decrementScrubCount(req->var, matls->get(m), neighbor);
                if (scrubout.active() && (req->var->getName() == dbgScrubVar || dbgScrubVar == "")
                    && (neighbor->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                  scrubout << Parallel::getMPIRank() << "   decrementing scrub count for requires of " << dws[dw]->getID() << "/"
                           << neighbor->getID() << "/" << matls->get(m) << "/" << req->var->getName() << ": " << count
                           << (count == 0 ? " - scrubbed\n" : "\n");
                }
              }
              catch (UnknownVariable& e) {
                std::cout << "   BAD BOY FROM Task : " << *this << " scrubbing " << *req << " PATCHES: " << *patches.get_rep() << std::endl;
                throw e;
              }
            }
          }
        }
      }
    }
  }  // end for req

  // Scrub modifies
  for (const Task::Dependency* mod = task->getModifies(); mod != 0; mod = mod->next) {
    int dw = mod->mapDataWarehouse();
    DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
    if (scrubmode == DataWarehouse::ScrubComplete
        || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(mod->var) == initialRequires.end())) {

      if (unscrubbables.find(mod->var->getName()) != unscrubbables.end())
        continue;

      constHandle<PatchSubset> patches = mod->getPatchesUnderDomain(getPatches());
      constHandle<MaterialSubset> matls = mod->getMaterialsUnderDomain(getMaterials());
      TypeDescription::Type type = mod->var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            int count = dws[dw]->decrementScrubCount(mod->var, matls->get(m), patch);
            if (scrubout.active() && (mod->var->getName() == dbgScrubVar || dbgScrubVar == "")
                && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1))
              scrubout << Parallel::getMPIRank() << "   decrementing scrub count for modifies of " << dws[dw]->getID() << "/"
              << patch->getID() << "/" << matls->get(m) << "/" << mod->var->getName() << ": " << count
              << (count == 0 ? " - scrubbed\n" : "\n");
          }
        }
      }
    }
  }

  // Set the scrub count for each of the computes variables
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
    TypeDescription::Type type = comp->var->typeDescription()->getType();
    if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
      int whichdw = comp->whichdw;
      int dw = comp->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if (scrubmode == DataWarehouse::ScrubComplete
          || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(comp->var) == initialRequires.end())) {
        constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(getMaterials());

        if (unscrubbables.find(comp->var->getName()) != unscrubbables.end()) {
          continue;
        }

        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            int matl = matls->get(m);
            int count;
            if (m_task_group->getScrubCount(comp->var, matl, patch, whichdw, count)) {
              if (scrubout.active() && (comp->var->getName() == dbgScrubVar || dbgScrubVar == "")
                  && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                scrubout << Parallel::getMPIRank() << "   setting scrub count for computes of " << dws[dw]->getID() << "/"
                         << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << ": " << count << '\n';
              }
              dws[dw]->setScrubCount(comp->var, matl, patch, count);
            }
            else {
              // Not in the scrub map, must be never needed...
              if (scrubout.active() && (comp->var->getName() == dbgScrubVar || dbgScrubVar == "")
                  && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                scrubout << Parallel::getMPIRank() << "   trashing variable immediately after compute: " << dws[dw]->getID() << "/"
                         << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << '\n';
              }
              dws[dw]->scrub(comp->var, matl, patch);
            }
          }
        }
      }
    }
  }
}  // end scrub()

//_____________________________________________________________________________
//
void DetailedTask::findRequiringTasks( const VarLabel*            var,
                                  std::list<DetailedTask*>& requiringTasks )
{
  // find requiring tasks

  // find external requires
  for (DependencyBatch* batch = getComputes(); batch != 0; batch = batch->m_comp_next) {
    for (DetailedDependency* dep = batch->m_head; dep != 0; dep = dep->m_next) {
      if (dep->m_req->var == var) {
        requiringTasks.insert(requiringTasks.end(), dep->m_to_tasks.begin(), dep->m_to_tasks.end());
      }
    }
  }

  // find internal requires
  std::map<DetailedTask*, InternalDependency*>::iterator internalDepIter;
  for (internalDepIter = m_internal_dependents.begin(); internalDepIter != m_internal_dependents.end(); ++internalDepIter) {
    if (internalDepIter->second->m_var_labels.find(var) != internalDepIter->second->m_var_labels.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::addComputes( DependencyBatch* comp )
{
  comp->m_comp_next = m_comp_head;
  m_comp_head = comp;
}

bool DetailedTask::addRequires( DependencyBatch* req )
{
  // return true if it is adding a new batch
  return m_requires.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
void DetailedTask::addInternalComputes(DependencyBatch* comp)
{
  comp->m_comp_next = m_internal_comp_head;
  m_internal_comp_head = comp;
}

//_____________________________________________________________________________
//
bool DetailedTask::addInternalRequires(DependencyBatch* req)
{
  // return true if it is adding a new batch
  return m_internal_requires.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
// can be called in one of two places - when the last MPI Recv has completed, or from MPIScheduler
void DetailedTask::checkExternalDepCount()
{
  DOUT(mpidbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " external deps: " << m_external_dependency_count
                                << " internal deps: " << m_num_pending_internal_dependencies);

  if (m_external_dependency_count == 0 && m_task_group->m_scheduler->useInternalDeps() && m_initiated && !m_task->usesMPI()) {
    m_task_group->m_mpi_completed_queue_lock.lock();

    DOUT(mpidbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
                                  << " MPI requirements satisfied, placing into external ready queue");
    if (m_externally_ready == false) {
      m_task_group->m_mpi_completed_tasks.push(this);
      m_externally_ready = true;
    }
    m_task_group->m_mpi_completed_queue_lock.unlock();
  }
}

//_____________________________________________________________________________
//
void DetailedTask::resetDependencyCounts()
{
  m_external_dependency_count = 0;
  m_externally_ready = false;
  m_initiated = false;

  m_num_pending_internal_dependencies = m_internal_dependencies.size();

  m_wait_timer.reset();
  m_exec_timer.reset();
}

void DetailedTask::addInternalDependency(       DetailedTask* prerequisiteTask,
                                     const VarLabel*     var )
{
  if (m_task_group->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    std::map<DetailedTask*, InternalDependency*>::iterator foundIt = prerequisiteTask->m_internal_dependents.find(this);
    if (foundIt == prerequisiteTask->m_internal_dependents.end()) {
      m_internal_dependencies.push_back(InternalDependency(prerequisiteTask, this, var, 0/* not satisfied */));
      prerequisiteTask->m_internal_dependents[this] = &m_internal_dependencies.back();
      m_num_pending_internal_dependencies = m_internal_dependencies.size();
    }
    else {
      foundIt->second->addVarLabel(var);
    }
  }
}


#ifdef HAVE_CUDA

void DetailedTask::assignDevice( unsigned int device )
{
  deviceNum_ = device;
  deviceNums_.insert ( device );
}

//unsigned int DetailedTask::getDeviceNum() const
//{
//  return deviceNum_;
//}

//For tasks where there are multiple devices for the task (i.e. data archiver output tasks)
std::set<unsigned int> DetailedTask::getDeviceNums() const
{
  return deviceNums_;
}

/*
cudaStream_t* DetailedTask::getCudaStreamForThisTask() const
{
  return getThisTasksCudaStream(0);
}*/

cudaStream_t* DetailedTask::getCudaStreamForThisTask(unsigned int deviceNum) const
{
  std::map <unsigned int, cudaStream_t*>::const_iterator it;
  it = d_cudaStreams.find(deviceNum);
  if (it != d_cudaStreams.end()) {
    return it->second;
  }
  return NULL;
}


//void DetailedTask::setCudaStreamForThisTask(cudaStream_t* s)
//{
//  //d_cudaStream = s;
//  setCudaStreamForThisTask(0, s);
//};

void DetailedTask::setCudaStreamForThisTask(unsigned int deviceNum, cudaStream_t* s)
{
  if (s == NULL) {
    printf("ERROR! - DetailedTask::setCudaStreamForThisTask() - A request was made to assign a stream at address NULL into this task %s\n", getName().c_str());
    SCI_THROW(InternalError("A request was made to assign a stream at address NULL into this task :"+ getName() , __FILE__, __LINE__));
  } else {
    if (d_cudaStreams.find(deviceNum) == d_cudaStreams.end()) {
      d_cudaStreams.insert(std::pair<unsigned int, cudaStream_t*>(deviceNum,s));
    } else {
      printf("ERROR! - DetailedTask::setCudaStreamForThisTask() - This task %s already had a stream assigned for device %d\n", getName().c_str(), deviceNum);
      SCI_THROW(InternalError("Detected CUDA kernel execution failure on task: "+ getName() , __FILE__, __LINE__));

    }
  }
};

void DetailedTask::clearCudaStreamsForThisTask() {
  d_cudaStreams.clear();
}
/*
bool DetailedTask::checkCudaStreamDoneForThisTask() const
{
  //Check all
  cudaError_t retVal;
  for (std::map<unsigned int, cudaStream_t*>::const_iterator it = d_cudaStreams.begin(); it != d_cudaStreams.end(); ++it) {
    OnDemandDataWarehouse::uintahSetCudaDevice(it->first);
    if (it->second == NULL) {
      printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Stream pointer with NULL address for task %s\n", getName().c_str());
      SCI_THROW(InternalError("Stream pointer with NULL address for task: " + getName() , __FILE__, __LINE__));
      return false;
    }
    retVal = cudaStreamQuery(*(it->second));
    if (retVal == cudaSuccess) {
    //  cout << "checking cuda stream " << d_cudaStream << "ready" << endl;
      continue;
    } else if (retVal == cudaErrorNotReady ) {

      retVal = cudaStreamQuery(*(it->second));
      return false;
    }
    else if (retVal ==  cudaErrorLaunchFailure) {
      printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - CUDA kernel execution failure on Task: %s\n", getName().c_str());
      SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task:"+ getName() , __FILE__, __LINE__));
      return false;
    } else { //other error
      printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - The stream %p had this error code %d.  This could mean that something else in the stream just hit an error.\n",  it->second, retVal);
      SCI_THROW(InternalError("ERROR! - Invalid stream query", __FILE__, __LINE__));
      return false;
    }

  }
  return true;
}
*/

bool DetailedTask::checkCudaStreamDoneForThisTask(unsigned int deviceNum_) const
{

  // sets the CUDA context, for the call to cudaEventQuery()
  cudaError_t retVal;
  OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum_);
  std::map<unsigned int, cudaStream_t*>::const_iterator it= d_cudaStreams.find(deviceNum_);
  if (it == d_cudaStreams.end()) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Request for stream information for device %d, but this task wasn't assigned any streams for this device.  For task %s\n", deviceNum_,  getName().c_str());
    SCI_THROW(InternalError("Request for stream information for a device, but it wasn't assigned any streams for that device.  For task: " + getName() , __FILE__, __LINE__));
    return false;
  }
  if (it->second == NULL) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Stream pointer with NULL address for task %s\n", getName().c_str());
    SCI_THROW(InternalError("Stream pointer with NULL address for task: " + getName() , __FILE__, __LINE__));
    return false;
  }

  retVal = cudaStreamQuery(*(it->second));
  if (retVal == cudaSuccess) {
//  cout << "checking cuda stream " << d_cudaStream << "ready" << endl;
    return true;
  }
  else if (retVal == cudaErrorNotReady ) {
    return false;
  }
  else if (retVal ==  cudaErrorLaunchFailure) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask(%d) - CUDA kernel execution failure on Task: %s\n", deviceNum_, getName().c_str());
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: " + getName() , __FILE__, __LINE__));
    return false;
  } else { //other error
    printf("Waiting for 60\n");
    Time::waitFor( (double)60 );
    CUDA_RT_SAFE_CALL (retVal);
    return false;
  }
}

bool DetailedTask::checkAllCudaStreamsDoneForThisTask() const
{
  // sets the CUDA context, for the call to cudaEventQuery()
  bool retVal = false;

  for (std::map<unsigned int ,cudaStream_t*>::const_iterator it=d_cudaStreams.begin(); it!=d_cudaStreams.end(); ++it){
    retVal = checkCudaStreamDoneForThisTask(it->first);
    if (retVal == false) {
      return retVal;
    }
  }

  return true;
}

void DetailedTask::setTaskGpuDataWarehouse(const unsigned int whichDevice, Task::WhichDW DW, GPUDataWarehouse* TaskDW ) {

  std::map<unsigned int, TaskGpuDataWarehouses>::iterator it;
  it = TaskGpuDWs.find(whichDevice);
  if (it != TaskGpuDWs.end()) {
    it->second.TaskGpuDW[DW] = TaskDW;

  } else {
    TaskGpuDataWarehouses temp;
    temp.TaskGpuDW[0] = NULL;
    temp.TaskGpuDW[1] = NULL;
    temp.TaskGpuDW[DW] = TaskDW;
    TaskGpuDWs.insert(std::pair<unsigned int, TaskGpuDataWarehouses>(whichDevice, temp));
  }
}

GPUDataWarehouse* DetailedTask::getTaskGpuDataWarehouse(const unsigned int whichDevice, Task::WhichDW DW) {
  std::map<unsigned int, TaskGpuDataWarehouses>::iterator it;
  it = TaskGpuDWs.find(whichDevice);
  if (it != TaskGpuDWs.end()) {
    return it->second.TaskGpuDW[DW];
  }
  return NULL;

}

void DetailedTask::deleteTaskGpuDataWarehouses() {
  for (std::map<unsigned int, TaskGpuDataWarehouses>::iterator it = TaskGpuDWs.begin(); it != TaskGpuDWs.end(); ++it) {
    for (int i = 0; i < 2; i++) {
        if (it->second.TaskGpuDW[i] != NULL) {
          //Note: Do not call the clear() method.  The Task GPU DWs only contains a "snapshot"
          //of the things in the GPU.  The host side GPU DWs is responsible for
          //deallocating all the GPU resources.  The only thing we do want to clean
          //up is that this GPUDW lives on the GPU.
          it->second.TaskGpuDW[i]->deleteSelfOnDevice();

          //void * getPlacementNewBuffer = it->second.TaskGpuDW[i]->getPlacementNewBuffer();

          //it->second.TaskGpuDW[i]->~GPUDataWarehouse();
          //free(getPlacementNewBuffer);

          it->second.TaskGpuDW[i]->cleanup();
          free(it->second.TaskGpuDW[i]);
          it->second.TaskGpuDW[i] = NULL;
        }
      }
  }
}


void DetailedTask::clearPreparationCollections(){

  deviceVars.clear();
  ghostVars.clear();
  taskVars.clear();
  varsToBeGhostReady.clear();
  varsBeingCopiedByTask.clear();
}
#endif // HAVE_CUDA

//_____________________________________________________________________________
//
void DetailedTask::done( std::vector<OnDemandDataWarehouseP>& dws )
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  int cnt = 1000;
  std::map<DetailedTask*, InternalDependency*>::iterator iter;
  for (iter = m_internal_dependents.begin(); iter != m_internal_dependents.end(); iter++) {
    InternalDependency* dep = (*iter).second;

    if (internaldbg.active()) {
      internaldbg << Parallel::getMPIRank() << " Dependency satisfied between " << *dep->m_dependent_task << " and " << *this << "\n";
    }

    dep->m_dependent_task->dependencySatisfied(dep);
    cnt++;
  }

  m_exec_timer.stop();
}

//_____________________________________________________________________________
//
void DetailedTask::dependencySatisfied( InternalDependency* dep )
{
  m_internal_dependency_lock.lock();
  {
    ASSERT(m_num_pending_internal_dependencies > 0);
    unsigned long currentGeneration = m_task_group->getCurrentDependencyGeneration();

    // if false, then the dependency has already been satisfied
    ASSERT(dep->m_satisfied_generation < currentGeneration);

    dep->m_satisfied_generation = currentGeneration;
    --m_num_pending_internal_dependencies;

    if (m_num_pending_internal_dependencies == 0) {
      m_task_group->internalDependenciesSatisfied(this);
    }
  }
  m_internal_dependency_lock.unlock();
}

//_____________________________________________________________________________
//
std::ostream& operator<<( std::ostream& out, const DetailedTask& task )
{
  coutLock.lock();
  {
    out << task.getTask()->getName();
    const PatchSubset* patches = task.getPatches();
    if (patches) {

      out << ", on patch";
      if (patches->size() > 1) {
        out << "es";
      }
      out << " ";
      for (int i = 0; i < patches->size(); i++) {
        if (i > 0) {
          out << ",";
        }
        out << patches->get(i)->getID();
      }
      // a once-per-proc task is liable to have multiple levels, and thus calls to getLevel(patches) will fail
      if (task.getTask()->getType() == Task::OncePerProc) {
        out << ", on multiple levels";
      }
      else {
        out << ", Level " << getLevel(patches)->getIndex();
      }
    }
    const MaterialSubset* matls = task.getMaterials();
    if (matls) {
      out << ", on material";
      if (matls->size() > 1) {
        out << "s";
      }
      out << " ";
      for (int i = 0; i < matls->size(); i++) {
        if (i > 0) {
          out << ",";
        }
        out << matls->get(i);
      }
    }
    out << ", resource (rank): ";
    if (task.getAssignedResourceIndex() == -1) {
      out << "unassigned";
    }
    else {
      out << task.getAssignedResourceIndex();
    }
  }
  coutLock.unlock();

  return out;
}

} // namespace Uintah

#endif // UINTAH_USING_EXPERIMENTAL
