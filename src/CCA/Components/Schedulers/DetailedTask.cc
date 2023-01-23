/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/DependencyBatch.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

#if defined(HAVE_GPU)
  #include <CCA/Components/Schedulers/GPUMemoryPool.h>
#endif

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParams.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/config_defs.h>

#include <sstream>
#include <string>


using namespace Uintah;

#if defined(HAVE_GPU)
extern Uintah::MasterLock cerrLock;

namespace {

  Uintah::MasterLock g_GridVarSuperPatch_mutex{};   // An ugly hack to get superpatches for host levels to work.

}
#endif

// declared in DetailedTasks.cc - used in both places to protect external ready queue (hence, extern here)
namespace Uintah {
  extern DebugStream gpu_stats;

  extern Uintah::MasterLock g_external_ready_mutex;
  extern Dout               g_scrubbing_dbg;
  extern std::string        g_var_scrub_dbg;
  extern int                g_patch_scrub_dbg;
}


namespace {
  Uintah::MasterLock g_internal_dependency_mutex{};
  Uintah::MasterLock g_dtask_output_mutex{};

  Dout g_internal_deps_dbg( "InternalDeps", "DetailedTask", "info on internal (intra-nodal) data dependencies", false);
  Dout g_external_deps_dbg( "ExternalDeps", "DetailedTask", "info on external (inter-nodal) data dependencies", false);
}


//_____________________________________________________________________________
//
DetailedTask::DetailedTask(       Task           * task
                          , const PatchSubset    * patches
                          , const MaterialSubset * matls
                          ,       DetailedTasks  * taskGroup
                          )
  : m_task( task )
  , m_patches( patches )
  , m_matls( matls )
  , m_task_group( taskGroup )
{
#if defined(HAVE_GPU)
  varLock = new Uintah::MasterLock{};
#endif
  if (m_patches) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(m_patches->getVector().begin(), m_patches->getVector().end(), Patch::Compare()));
    m_patches->addReference();
  }
  if (m_matls) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(m_matls->getVector().begin(), m_matls->getVector().end()));
    m_matls->addReference();
  }
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
#ifdef USE_KOKKOS_INSTANCE
  clearKokkosInstancesForThisTask();
#endif
}

//_____________________________________________________________________________
//
void
DetailedTask::doit( const ProcessorGroup                      * pg
                  ,       std::vector<OnDemandDataWarehouseP> & oddws
                  ,       std::vector<DataWarehouseP>         & dws
                  ,       CallBackEvent                         event // = CallBackEvent::CPU
                  )
{
  // Stop timing the task wait
  m_wait_timer.stop();

  // Start timing the execution duration
  m_exec_timer.start();

  if ( g_internal_deps_dbg ) {
    std::ostringstream message;
    message << "DetailedTask " << this << " begin doit()\n";
    message << " task is " << m_task << "\n";
    message << "   num Pending Deps: " << m_num_pending_internal_dependencies << "\n";
    message << "   Originally needed deps (" << m_internal_dependencies.size() << "):\n";

    auto iter = m_internal_dependencies.begin();
    for (size_t i = 0u; iter != m_internal_dependencies.end(); ++iter, ++i) {
      message << i << ":    " << *((*iter).m_prerequisite_task->getTask()) << "\n";
    }

    DOUT(true, message.str());
  }

  for (size_t i = 0; i < dws.size(); ++i) {
    if (oddws[i] != nullptr) {
      oddws[i]->pushRunningTask(m_task, &oddws);
    }
  }

  // Start loading up the UintahParams object
  UintahParams uintahParams;
  uintahParams.setTaskIntPtr(reinterpret_cast<intptr_t>(this));
  uintahParams.setProcessorGroup(pg);
  uintahParams.setCallBackEvent(event);

#if defined(HAVE_GPU)
  // Load in streams whether this is a CPU task or GPU task. GPU tasks
  // need streams. CPU tasks can also use streams for D2H copies or
  // transferFrom calls.
#ifdef TASK_MANAGES_EXECSPACE
  // Now in the Task
#else
  int numStreams = m_task->maxStreamsPerTask();
  for (int i = 0; i < numStreams; i++) {
    cudaStream_t* stream = this->getCudaStreamForThisTask(i);
    uintahParams.setStream(stream);
  }
#endif
#endif

  // Determine if task will be executed on CPU or GPU
  if ( m_task->usesDevice() ) {
#if defined(HAVE_GPU)
    // Run the GPU task.  Technically the engine has structure to run
    // one task on multiple devices if that task had patches on
    // multiple devices.  So run the task once per device.  As often
    // as possible, we want to design tasks so each task runs on only
    // once device, instead of a one to many relationship.

    //for (deviceNumSetIter deviceNums_it = deviceNums_.begin(); deviceNums_it != deviceNums_.end(); ++deviceNums_it) {
    // const unsigned int currentDevice = *deviceNums_it;
      int currentDevice = 0;
      // Base call is commented out
      // OnDemandDataWarehouse::uintahSetCudaDevice(currentDevice);
      GPUDataWarehouse* host_oldtaskdw = getTaskGpuDataWarehouse(currentDevice, Task::OldDW);
      GPUDataWarehouse* device_oldtaskdw = nullptr;
      if (host_oldtaskdw) {
        device_oldtaskdw = host_oldtaskdw->getdevice_ptr();
      }
      GPUDataWarehouse* host_newtaskdw = getTaskGpuDataWarehouse(currentDevice, Task::NewDW);
      GPUDataWarehouse* device_newtaskdw = nullptr;
      if (host_newtaskdw) {
        device_newtaskdw = host_newtaskdw->getdevice_ptr();
      }

      // Load up the uintahParams object with task data warehouses (if
      // they are needed)
      uintahParams.setDetailedTask(this);
      uintahParams.setTaskGpuDWs(device_oldtaskdw, device_newtaskdw);

      m_task->doit( m_patches, m_matls, dws, uintahParams );
   //}
#else
    SCI_THROW(InternalError("A task was marked as GPU enabled, but Uintah was not compiled for CUDA support", __FILE__, __LINE__));
#endif

  } else {
    m_task->doit( m_patches, m_matls, dws, uintahParams );
  }

  for (size_t i = 0u; i < dws.size(); ++i) {
    if ( oddws[i] != nullptr ) {
//     oddws[i]->checkTasksAccesses( d_patches, d_matls );
      oddws[i]->popRunningTask();
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::scrub( std::vector<OnDemandDataWarehouseP> & dws )
{
  DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Starting scrub after task: " << *this);

  const Task* task = getTask();

  const std::set<const VarLabel*, VarLabel::Compare> & initialRequires = m_task_group->getSchedulerCommon()->getInitialRequiredVars();
  const std::set<std::string>                        &   unscrubbables = m_task_group->getSchedulerCommon()->getNoScrubVars();

  // Decrement the scrub count for each of the required variables
  for (const Task::Dependency* req = task->getRequires(); req != nullptr; req = req->m_next) {
    TypeDescription::Type type = req->m_var->typeDescription()->getType();
    Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
    if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
      int dw = req->mapDataWarehouse();

      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if (scrubmode == DataWarehouse::ScrubComplete
          || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(req->m_var) == initialRequires.end())) {

        if (unscrubbables.find(req->m_var->getName()) != unscrubbables.end())
          continue;

        constHandle<PatchSubset> patches = req->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(getMaterials());
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          Patch::selectType neighbors;
          IntVector low, high;

          if (req->m_patches_dom == Task::CoarseLevel || req->m_patches_dom == Task::FineLevel || req->m_num_ghost_cells == 0) {
            // we already have the right patches
            neighbors.push_back(patch);
          }
          else {
            patch->computeVariableExtents(type, req->m_var->getBoundaryLayer(), req->m_gtype, req->m_num_ghost_cells, neighbors, low, high);
          }

          for (unsigned int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];

            if ( req->m_patches_dom == Task::ThisLevel && patch != neighbor ) {
              // Don't scrub on AMR overlapping patches...
              IntVector l = Max(neighbor->getExtraLowIndex(basis, req->m_var->getBoundaryLayer()), low);
              IntVector h = Min(neighbor->getExtraHighIndex(basis, req->m_var->getBoundaryLayer()), high);

              patch->cullIntersection(basis, req->m_var->getBoundaryLayer(), neighbor->getRealPatch(), l, h);

              if (l == h){
                continue;
              }
            }

            if (req->m_patches_dom == Task::FineLevel) {
              // Don't count if it only overlaps extra cells
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
                // there are a few rare cases in an AMR framework
                // where you require from an OldDW, but only ones
                // internal to the W-cycle (and not the previous
                // timestep) which can have variables not exist in the
                // OldDW.
                count = dws[dw]->decrementScrubCount(req->m_var, matls->get(m), neighbor);
                if (g_scrubbing_dbg && (req->m_var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") &&
                                       (neighbor->getID() == g_patch_scrub_dbg   || g_patch_scrub_dbg == -1)) {
                  DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << "   decrementing scrub count for requires of " << dws[dw]->getID() << "/"
                                         << neighbor->getID() << "/" << matls->get(m) << "/" << req->m_var->getName() << ": " << count
                                         << (count == 0 ? " - scrubbed\n" : "\n"));
                }
              }
              catch (UnknownVariable& e) {
                std::cerr << "   BAD BOY FROM Task : " << *this << " scrubbing " << *req << " PATCHES: " << *patches.get_rep() << std::endl;
                throw e;
              }
            }
          }
        }
      }
    }
  }  // end for req

  // Scrub modifies
  for (const Task::Dependency* mod = task->getModifies(); mod != nullptr; mod = mod->m_next) {
    int dw = mod->mapDataWarehouse();
    DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
    if (scrubmode == DataWarehouse::ScrubComplete
        || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(mod->m_var) == initialRequires.end())) {

      if (unscrubbables.find(mod->m_var->getName()) != unscrubbables.end())
        continue;

      constHandle<PatchSubset> patches = mod->getPatchesUnderDomain(getPatches());
      constHandle<MaterialSubset> matls = mod->getMaterialsUnderDomain(getMaterials());
      TypeDescription::Type type = mod->m_var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            int count = dws[dw]->decrementScrubCount(mod->m_var, matls->get(m), patch);
            if (g_scrubbing_dbg && (mod->m_var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") &&
                                   (patch->getID() == g_patch_scrub_dbg || g_patch_scrub_dbg == -1)) {
              DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << "   decrementing scrub count for modifies of " << dws[dw]->getID() << "/"
                                     << patch->getID() << "/" << matls->get(m) << "/" << mod->m_var->getName() << ": " << count
                                     << (count == 0 ? " - scrubbed\n" : "\n"));
            }
          }
        }
      }
    }
  }

  // Set the scrub count for each of the computes variables
  for (const Task::Dependency* comp = task->getComputes(); comp != nullptr; comp = comp->m_next) {
    TypeDescription::Type type = comp->m_var->typeDescription()->getType();
    if (type != TypeDescription::ReductionVariable && type != TypeDescription::SoleVariable) {
      int whichdw = comp->m_whichdw;
      int dw = comp->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if (scrubmode == DataWarehouse::ScrubComplete
          || (scrubmode == DataWarehouse::ScrubNonPermanent && initialRequires.find(comp->m_var) == initialRequires.end())) {
        constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(getMaterials());

        if (unscrubbables.find(comp->m_var->getName()) != unscrubbables.end()) {
          continue;
        }

        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            int matl = matls->get(m);
            int count;
            if ( m_task_group->getScrubCount(comp->m_var, matl, patch, whichdw, count) ) {
              if (g_scrubbing_dbg && (comp->m_var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") &&
                                     (patch->getID() == g_patch_scrub_dbg || g_patch_scrub_dbg == -1)) {
                DOUT(true, "Rank-" << Parallel::getMPIRank() << "   setting scrub count for computes of " << dws[dw]->getID() << "/"
                                   << patch->getID() << "/" << matls->get(m) << "/" << comp->m_var->getName() << ": " << count);
              }
              dws[dw]->setScrubCount(comp->m_var, matl, patch, count);
            }
            else {
              // Not in the scrub map, must be never needed...
              if (g_scrubbing_dbg && (comp->m_var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") &&
                                     (patch->getID() == g_patch_scrub_dbg || g_patch_scrub_dbg == -1)) {
                DOUT(true, "Rank-" << Parallel::getMPIRank() << "   trashing variable immediately after compute: " << dws[dw]->getID() << "/"
                                   << patch->getID() << "/" << matls->get(m) << "/" << comp->m_var->getName());
              }
              dws[dw]->scrub(comp->m_var, matl, patch);
            }
          }
        }
      }
    }
  }
}  // end scrub()

//_____________________________________________________________________________
//
void
DetailedTask::findRequiringTasks( const VarLabel                 * var
                                ,       std::list<DetailedTask*> & requiringTasks
                                )
{
  // find external requires
  for (DependencyBatch* batch = getComputes(); batch != nullptr; batch = batch->m_comp_next) {
    for (DetailedDep* dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
      if (dep->m_req->m_var == var) {
        requiringTasks.insert(requiringTasks.end(), dep->m_to_tasks.begin(), dep->m_to_tasks.end());
      }
    }
  }

  // find internal requires
  std::map<DetailedTask*, InternalDependency*>::iterator internalDepIter;
  for (internalDepIter = m_internal_dependents.begin(); internalDepIter != m_internal_dependents.end(); ++internalDepIter) {
    if (internalDepIter->second->m_vars.find(var) != internalDepIter->second->m_vars.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::addComputes( DependencyBatch * comp )
{
  comp->m_comp_next = m_comp_head;
  m_comp_head = comp;
}

//_____________________________________________________________________________
//
bool
DetailedTask::addRequires( DependencyBatch * req )
{
  // return true if it is adding a new batch
  return m_reqs.insert( std::make_pair( req, req ) ).second;
}

//_____________________________________________________________________________
//
void
DetailedTask::addInternalComputes( DependencyBatch * comp )
{
  comp->m_comp_next = m_internal_comp_head;
  m_internal_comp_head = comp;
}

//_____________________________________________________________________________
//
bool
DetailedTask::addInternalRequires( DependencyBatch * req )
{
  // return true if it is adding a new batch
  return m_internal_reqs.insert( std::make_pair(req, req) ).second;
}

//_____________________________________________________________________________
// can be called in one of two places - when the last MPI Recv has
// completed, or from MPIScheduler
void
DetailedTask::checkExternalDepCount()
{
  std::lock_guard<Uintah::MasterLock> external_ready_guard(g_external_ready_mutex);

  DOUT(g_external_deps_dbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " external deps: "
                                    << m_external_dependency_count.load(std::memory_order_acquire)
                                    << " internal deps: " << m_num_pending_internal_dependencies);

  if ((m_external_dependency_count.load(std::memory_order_acquire) == 0) && m_task_group->m_sched_common->useInternalDeps() &&
       m_initiated.load(std::memory_order_acquire) && !m_task->usesMPI()) {

    DOUT(g_external_deps_dbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
                                      << " MPI requirements satisfied, placing into external ready queue");


    unsigned int amountTaskNameExpectedToRun = Uintah::Parallel::getAmountTaskNameExpectedToRun();

    if (amountTaskNameExpectedToRun > 0) {
      std::string task_to_debug_name = Uintah::Parallel::getTaskNameToTime();
      std::string current_task = this->getTask()->getName();
      if ( current_task.size() >= task_to_debug_name.size() && this->getTask() && this->getTask()->getName().substr(0, task_to_debug_name.size()) == task_to_debug_name ) {
        m_task_group->atomic_task_to_debug_size.fetch_add(1);
      }
    }

    if (m_externally_ready.load(std::memory_order_acquire) == false) {
      m_task_group->m_mpi_completed_tasks.push(this);
      m_task_group->m_atomic_mpi_completed_tasks_size.fetch_add(1);
      m_externally_ready.store(true, std::memory_order_release);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::resetDependencyCounts()
{
  m_external_dependency_count.store(     0, std::memory_order_relaxed);
  m_externally_ready.store(          false, std::memory_order_relaxed);
  m_initiated.store(                 false, std::memory_order_relaxed);

  m_wait_timer.reset(true);
  m_exec_timer.reset(true);
}

//_____________________________________________________________________________
//
void
DetailedTask::addInternalDependency(       DetailedTask * prerequisiteTask
                                   , const VarLabel     * var
                                   )
{
  if ( m_task_group->mustConsiderInternalDependencies() ) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    std::map<DetailedTask*, InternalDependency*>::iterator foundIt = prerequisiteTask->m_internal_dependents.find(this);
    if (foundIt == prerequisiteTask->m_internal_dependents.end()) {
      m_internal_dependencies.push_back(InternalDependency(prerequisiteTask, this, var, 0 /* 0 == not satisfied */));
      prerequisiteTask->m_internal_dependents[this] = &m_internal_dependencies.back();
      m_num_pending_internal_dependencies = m_internal_dependencies.size();

      DOUT(g_internal_deps_dbg, "Rank-" << Parallel::getMPIRank() << " Adding dependency between " << *this << " and " << *prerequisiteTask << " for var " << var->getName()
                                        << " source dep count: " << m_num_pending_internal_dependencies << " pre-req dep count " << prerequisiteTask->m_internal_dependents.size());
    }
    else {
      foundIt->second->addVarLabel(var);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::done( std::vector<OnDemandDataWarehouseP> & dws )
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  if (g_internal_deps_dbg) {
    std::ostringstream message;
    message << "This: " << this << " is done with task: " << m_task << "\n";
    message << "Name is: " << m_task->getName() << " which has (" << m_internal_dependents.size() << ") tasks waiting on it:";
    DOUT( true, message.str() );
  }

  for (auto iter = m_internal_dependents.begin(); iter != m_internal_dependents.end(); ++iter) {
    InternalDependency* dep = (*iter).second;
    dep->m_dependent_task->dependencySatisfied(dep);

    DOUT(g_internal_deps_dbg, "Rank-" << Parallel::getMPIRank() << " Dependency satisfied between " << *dep->m_dependent_task << " and " << *this);
  }

  m_exec_timer.stop();
}

//_____________________________________________________________________________
//
void
DetailedTask::dependencySatisfied( InternalDependency * dep )
{
  std::lock_guard<Uintah::MasterLock> internal_dependency_guard(g_internal_dependency_mutex);

  ASSERT(m_num_pending_internal_dependencies > 0);
  unsigned long currentGeneration = m_task_group->getCurrentDependencyGeneration();

  // if false, then the dependency has already been satisfied
  ASSERT(dep->m_satisfied_generation < currentGeneration);

  dep->m_satisfied_generation = currentGeneration;
  m_num_pending_internal_dependencies--;

  DOUT(g_internal_deps_dbg, *(dep->m_dependent_task->getTask()) << " has " << m_num_pending_internal_dependencies << " left.");

  DOUT(g_internal_deps_dbg, "Rank-" << Parallel::getMPIRank() << " satisfying dependency: prereq: " << *dep->m_prerequisite_task
                                    << " dep: " << *dep->m_dependent_task << " numPending: " << m_num_pending_internal_dependencies);

  if (m_num_pending_internal_dependencies == 0) {
    m_task_group->internalDependenciesSatisfied(this);
    m_num_pending_internal_dependencies = m_internal_dependencies.size();  // reset for next timestep
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::emitEdges( ProblemSpecP edgesElement )
{
  for (auto req_iter = m_reqs.begin(); req_iter != m_reqs.end(); ++req_iter) {
    DetailedTask* fromTask = (*req_iter).first->m_from_task;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  for (auto iter = m_internal_dependencies.begin(); iter != m_internal_dependencies.end(); ++iter) {
    DetailedTask* fromTask = (*iter).m_prerequisite_task;
    if (getTask()->isReductionTask() && fromTask->getTask()->isReductionTask()) {
      // Ignore internal links between reduction tasks because they
      // are only needed for logistic reasons
      continue;
    }
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }
}

//_____________________________________________________________________________
//
class PatchIDIterator {

public:

  PatchIDIterator( const std::vector<const Patch*>::const_iterator& iter )
      : m_const_iter(iter)
  {}

  PatchIDIterator& operator=( const PatchIDIterator & iter2 )
  {
    m_const_iter = iter2.m_const_iter;
    return *this;
  }

  int operator*()
  {
    const Patch* patch = *m_const_iter;  //vector<Patch*>::iterator::operator*();
    return patch ? patch->getID() : -1;
  }

  PatchIDIterator & operator++()
  {
    m_const_iter++;
    return *this;
  }

  bool operator!=( const PatchIDIterator & iter2 )
  {
    return m_const_iter != iter2.m_const_iter;
  }


private:

  std::vector<const Patch*>::const_iterator m_const_iter;

};

//_____________________________________________________________________________
//
std::string
DetailedTask::getName() const
{
  if (m_name != "") {
    return m_name;
  }

  m_name = std::string( m_task->getName() );

  if( m_patches != nullptr ) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder( PatchIDIterator( m_patches->getVector().begin()), PatchIDIterator( m_patches->getVector().end() ) );
    m_name += std::string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if( m_matls != nullptr ) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder( m_matls->getVector().begin(), m_matls->getVector().end() );
    m_name += std::string(" (Matls: ") + matlSet.toString() + ")";
  }

  return m_name;
}


namespace Uintah {

std::ostream&
operator<<( std::ostream & out, const DetailedTask & dtask )
{
  g_dtask_output_mutex.lock();
  {
    out << dtask.getTask()->getName();
    const PatchSubset* patches = dtask.getPatches();
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
      if (dtask.getTask()->getType() == Task::OncePerProc || dtask.getTask()->getType() == Task::Hypre) {
        out << ", on multiple levels";
      }
      else {
        out << ", Level " << getLevel(patches)->getIndex();
      }
    }
    const MaterialSubset* matls = dtask.getMaterials();
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
    if (dtask.getAssignedResourceIndex() == -1) {
      out << "unassigned";
    }
    else {
      out << dtask.getAssignedResourceIndex();
    }
  }
  g_dtask_output_mutex.unlock();

  return out;
}

} // end namespace Uintah

#if defined(HAVE_GPU)

#ifdef TASK_MANAGES_EXECSPACE
  // Now in the Task
#else
//_____________________________________________________________________________
//
void
DetailedTask::assignDevice( unsigned int device_id )
{
  // m_deviceNum = device_id;
  m_deviceNums.insert( device_id );
}

//_____________________________________________________________________________
// For tasks where there are multiple devices for the task (i.e. data
// archiver output tasks)
DetailedTask::deviceNumSet
DetailedTask::getDeviceNums() const
{
  return m_deviceNums;
}

//_____________________________________________________________________________
//
cudaStream_t*
DetailedTask::getCudaStreamForThisTask( unsigned int device_id ) const
{
  cudaStreamMapIter it = m_cudaStreams.find(device_id);

  if (it != m_cudaStreams.end()) {
    return it->second;
  }

  return nullptr;
}

//_____________________________________________________________________________
//
void
DetailedTask::setCudaStreamForThisTask( unsigned int   device_id
                                      , cudaStream_t * stream
                                      )
{
  if (stream == nullptr) {
    printf("ERROR! - DetailedTask::setCudaStreamForThisTask() - "
           "A request was made to assign a stream at address nullptr "
           "into this task %s\n", getName().c_str());
    SCI_THROW(InternalError("A request was made to assign a stream at "
                            "address nullptr into this task :" +
                            getName() , __FILE__, __LINE__));
  } else {
    if (m_cudaStreams.find(device_id) == m_cudaStreams.end()) {
      m_cudaStreams.insert(std::pair<unsigned int, cudaStream_t*>(device_id, stream));
    } else {
      printf("ERROR! - DetailedTask::setCudaStreamForThisTask() - "
             "This task %s already had a stream assigned for device %d\n",
             getName().c_str(), device_id);
      SCI_THROW(InternalError("Detected CUDA kernel execution failure on task: " +
                              getName() , __FILE__, __LINE__));
    }
  }
};
#endif

#ifdef USE_KOKKOS_INSTANCE
//_____________________________________________________________________________
//
void
DetailedTask::clearKokkosInstancesForThisTask() {
  m_task->clearKokkosInstancesForThisTask(reinterpret_cast<intptr_t>(this));
}

//_____________________________________________________________________________
//
bool
DetailedTask::checkAllKokkosInstancesDoneForThisTask() const
{
  return m_task->checkAllKokkosInstancesDoneForThisTask(reinterpret_cast<intptr_t>(this));
}
#else
//_____________________________________________________________________________
//
void
DetailedTask::reclaimCudaStreamsIntoPool() {
#ifdef TASK_MANAGES_EXECSPACE
  m_task->reclaimCudaStreamsIntoPool(reinterpret_cast<intptr_t>(this));
#else
  // Once streams are reclaimed, clearCudaStreamsForThisTask is called.
  GPUMemoryPool::reclaimCudaStreamsIntoPool(this);
#endif
}

//_____________________________________________________________________________
//
void
DetailedTask::clearCudaStreamsForThisTask() {
#ifdef TASK_MANAGES_EXECSPACE
  m_task->clearCudaStreamsForThisTask(reinterpret_cast<intptr_t>(this));
#else
  m_cudaStreams.clear();
#endif
}

//_____________________________________________________________________________
//
bool
DetailedTask::checkCudaStreamDoneForThisTask( unsigned int device_id ) const
{
#ifdef TASK_MANAGES_EXECSPACE
  return m_task->checkCudaStreamDoneForThisTask(reinterpret_cast<intptr_t>(this), device_id);
#else
  // sets the CUDA context, for the call to cudaEventQuery()

  if (device_id != 0) {
   printf("Error, DetailedTask::checkCudaStreamDoneForThisTask is %u\n", device_id);
   exit(-1);
  }

  // Base call is commented out
  // OnDemandDataWarehouse::uintahSetCudaDevice(device_id);
  cudaStreamMapIter it = m_cudaStreams.find(device_id);

  // if (it == m_cudaStreams.end()) {
  //   printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Request for stream information for device %d, but this task wasn't assigned any streams for this device.  For task %s\n", device_id,  getName().c_str());
  //   SCI_THROW(InternalError("Request for stream information for a device, but it wasn't assigned any streams for that device.  For task: " + getName() , __FILE__, __LINE__));
  //   return false;
  // }
  // else if (it->second == nullptr) {
  //   printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Stream pointer with nullptr address for task %s\n", getName().c_str());
  //   SCI_THROW(InternalError("Stream pointer with nullptr address for task: " + getName() , __FILE__, __LINE__));
  //   return false;
  // }

  cudaError_t retVal = cudaStreamQuery(*(it->second));
  if (retVal == cudaSuccess) {
    return true;
  }
  else if (retVal == cudaErrorNotReady ) {
    return false;
  }
  else if (retVal ==  cudaErrorLaunchFailure) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask(%d) - CUDA kernel execution failure on Task: %s\n", device_id, getName().c_str());
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: " + getName() , __FILE__, __LINE__));
    return false;
  } else { //other error
    printf("\nA CUDA error occurred with error code %d.\n\n"
           "Waiting for 60 seconds\n", retVal);

    int sleepTime = 60;

    struct timespec ts;
    ts.tv_sec = (int) sleepTime;
    ts.tv_nsec = (int)(1.e9 * (sleepTime - ts.tv_sec));

    nanosleep(&ts, &ts);

    CUDA_RT_SAFE_CALL (retVal);
    return false;
  }
#endif
}

//_____________________________________________________________________________
//
bool
DetailedTask::checkAllCudaStreamsDoneForThisTask() const
{
#ifdef TASK_MANAGES_EXECSPACE
  return m_task->checkAllCudaStreamsDoneForThisTask(reinterpret_cast<intptr_t>(this));
#else
  // A task can have multiple streams (such as an output task pulling
  // from multiple GPUs).  Check all streams to see if they are done.
  // If any one stream isn't done, return false.  If nothing returned
  // false, then they all must be good to go.
  bool retVal = false;

  for (auto & it : m_cudaStreams) {
    retVal = checkCudaStreamDoneForThisTask(it.first);
    if (retVal == false) {
      return retVal;
    }
  }

  return true;
#endif
}
#endif

//_____________________________________________________________________________
//
void
DetailedTask::setTaskGpuDataWarehouse( const unsigned int       whichDevice
                                     ,       Task::WhichDW      DW
                                     ,       GPUDataWarehouse * TaskDW
                                     )
{
  auto iter = TaskGpuDWs.find(whichDevice);
  if (iter != TaskGpuDWs.end()) {
    iter->second.TaskGpuDW[DW] = TaskDW;

  } else {
    TaskGpuDataWarehouses temp;
    temp.TaskGpuDW[0]  = nullptr;
    temp.TaskGpuDW[1]  = nullptr;
    temp.TaskGpuDW[DW] = TaskDW;
    TaskGpuDWs.insert(std::pair<unsigned int, TaskGpuDataWarehouses>(whichDevice, temp));
  }
}

//_____________________________________________________________________________
//
GPUDataWarehouse*
DetailedTask::getTaskGpuDataWarehouse( const unsigned int  whichDevice
                                     ,       Task::WhichDW DW
                                     )
{
  auto iter = TaskGpuDWs.find(whichDevice);
  if (iter != TaskGpuDWs.end()) {
    return iter->second.TaskGpuDW[DW];
  }
  return nullptr;

}

//_____________________________________________________________________________
//
void
DetailedTask::deleteTaskGpuDataWarehouses()
{
  for (auto iter = TaskGpuDWs.begin(); iter != TaskGpuDWs.end(); ++iter) {
    for (int i = 0; i < 2; i++) {
        if (iter->second.TaskGpuDW[i] != nullptr) {
          // Note: Do not call the clear() method.  The Task GPU DWs
          // only contains a "snapshot" of the things in the GPU.  The
          // host side GPU DWs is responsible for deallocating all the
          // GPU resources.  The only thing we do want to clean up is
          // that this GPUDW lives on the GPU.
          iter->second.TaskGpuDW[i]->deleteSelfOnDevice();
          iter->second.TaskGpuDW[i]->cleanup();

          free(iter->second.TaskGpuDW[i]);
          iter->second.TaskGpuDW[i] = nullptr;
        }
      }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::clearPreparationCollections()
{
  deviceVars.clear();
  ghostVars.clear();
  taskVars.clear();
  varsToBeGhostReady.clear();
  varsBeingCopiedByTask.clear();
}

//_____________________________________________________________________________
//
void
DetailedTask::addTempHostMemoryToBeFreedOnCompletion( void * ptr )
{
  taskHostMemoryPoolItems.push(ptr);
}

//_____________________________________________________________________________
//
void
DetailedTask::addTempCudaMemoryToBeFreedOnCompletion( unsigned int   device_id
                                                    , void         * ptr
                                                    )
{
  gpuMemoryPoolDevicePtrItem gpuItem(device_id, ptr);
  taskCudaMemoryPoolItems.push_back(gpuItem);
}

//_____________________________________________________________________________
//
void
DetailedTask::deleteTemporaryTaskVars()
{
  // clean out the host list
  while (!taskHostMemoryPoolItems.empty()) {
    cudaHostUnregister(taskHostMemoryPoolItems.front());
    // TODO: Deletes a void*, and that doesn't call any object destructors
    delete[] taskHostMemoryPoolItems.front();
    taskHostMemoryPoolItems.pop();
  }

  // and the device
  for (auto p : taskCudaMemoryPoolItems) {
    GPUMemoryPool::freeCudaSpaceFromPool(p.device_id, p.ptr);
  }
  taskCudaMemoryPoolItems.clear();
}

//______________________________________________________________________
//
void
DetailedTask::prepareGpuDependencies( DependencyBatch       * batch
                                       , const VarLabel        * pos_var
                                       , OnDemandDataWarehouse * dw
                                       , OnDemandDataWarehouse * old_dw
                                       , const DetailedDep     * dep
                                       , DeviceVarDest           dest
                                       )
{

  // This should handle the following scenarios:

  // GPU -> different GPU same node (write to GPU array, move to other
  // device memory, copy in via copyGPUGhostCellsBetweenDevices)

  // GPU -> different GPU another node (write to GPU array, move to
  // host memory, copy via MPI)

  // GPU -> CPU another node (write to GPU array, move to host memory,
  // copy via MPI)

  // It should not handle
  // GPU -> CPU same node (handled in initateH2D)
  // GPU -> same GPU same node (handled in initateH2D)

  // This method therefore indicates that staging/contiguous arrays
  // are needed, and what ghost cell copies need to occur in the GPU.

  if (dep->isNonDataDependency()) {
    return;
  }

  const VarLabel* label = dep->m_req->m_var;
  const Patch* fromPatch = dep->m_from_patch;
  const int matlIndx = dep->m_matl;
  const Level* level = fromPatch->getLevel();
  const int levelID = level->getID();

  // TODO: Ask Alan about everything in the dep object.
  // the toTasks (will there be more than one?)
  // the dep->comp (computes?)
  // the dep->req (requires?)

  DetailedTask* toTask = nullptr;
  // Go through all toTasks
  for (std::list<DetailedTask*>::const_iterator iter = dep->m_to_tasks.begin(); iter != dep->m_to_tasks.end(); ++iter) {
    toTask = (*iter);

    constHandle<PatchSubset> patches = toTask->getPatches();
    const int numPatches = patches->size();
    //const Patch* toPatch = toTask->getPatches()->get(0);
    //if (toTask->getPatches()->size() > 1) {
    // printf("ERROR:\nDetailedTask::prepareGpuDependencies() does not yet support a dependency that has multiple destination patches\n");
    // SCI_THROW( InternalError("DetailedTask::prepareGpuDependencies() does not yet support a dependency that has multiple destination patches", __FILE__, __LINE__));
    //}
    for (int i = 0; i < numPatches; i++) {
      const Patch* toPatch = patches->get(i);
      const int fromresource = this->getAssignedResourceIndex();
      const int toresource = toTask->getAssignedResourceIndex();

      const int fromDeviceIndex = GpuUtilities::getGpuIndexForPatch(fromPatch);

      // for now, assume that task will only work on one device
      const int toDeviceIndex = GpuUtilities::getGpuIndexForPatch(toTask->getPatches()->get(0));

      if ((fromresource == toresource) && (fromDeviceIndex == toDeviceIndex)) {
        // don't handle GPU -> same GPU same node here
        continue;
      }

      GPUDataWarehouse* gpudw = nullptr;
      if (fromDeviceIndex != -1) {
        gpudw = dw->getGPUDW(fromDeviceIndex);
        if (!gpudw->isValidOnGPU(label->getName().c_str(), fromPatch->getID(), matlIndx, levelID)) {
          continue;
        }
      } else {
        SCI_THROW(
            InternalError("Device index not found for "+label->getFullName(matlIndx, fromPatch), __FILE__, __LINE__));
      }

      switch (label->typeDescription()->getType()) {
        case TypeDescription::ParticleVariable: {
        }
          break;
        case TypeDescription::NCVariable:
        case TypeDescription::CCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {

          // TODO, This compiles a list of regions we need to copy
          // into contiguous arrays.  We don't yet handle a scenario
          // where the ghost cell region is exactly the same size as
          // the variable, meaning we don't need to create an array
          // and copy to it.

          // We're going to copy the ghost vars from the source
          // variable (already in the GPU) to a destination array (not
          // yet in the GPU).  So make sure there is a destination.

          // See if we're already planning on making this exact copy.
          // If so, don't do it again.
          IntVector host_low, host_high, host_offset, host_size;
          host_low = dep->m_low;
          host_high = dep->m_high;
          host_offset = dep->m_low;
          host_size = dep->m_high - dep->m_low;
          const size_t elementDataSize = OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType());
          const size_t memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
          // If this staging var already exists, then assume the full
          // ghost cell copying information has already been set up
          // previously.  (Duplicate dependencies show up by this
          // point, so just ignore the duplicate).

          // TODO, this section should be treated atomically.
          // Duplicates do happen, and we don't yet handle if two of
          // the duplicates try to get added to
          // this->getDeviceVars().add() simultaneously.

          // NOTE: On the CPU, a ghost cell face may be sent from
          // patch A to patch B, while a ghost cell edge/line may be
          // sent from patch A to patch C, and the line of data for C
          // is wholly within the face data for B.  For the sake of
          // preparing for cuda aware MPI, we still want to create two
          // staging vars here, a contiguous face for B, and a
          // contiguous edge/line for C.
          if (!(this->getDeviceVars().stagingVarAlreadyExists(dep->m_req->m_var, fromPatch, matlIndx, levelID, host_low, host_size, dep->m_req->mapDataWarehouse()))) {


            // TODO: This host var really should be created last
            // minute only if it's copying data to host.  Not here.
            // TODO: Verify this cleans up.  If so change the comment.
            GridVariableBase* tempGhostVar = dynamic_cast<GridVariableBase*>(label->typeDescription()->createInstance());
            tempGhostVar->allocate(dep->m_low, dep->m_high);

            // Indicate we want a staging array in the device.
            this->getDeviceVars().add(fromPatch, matlIndx, levelID, true, host_size, memSize, elementDataSize,
                                         host_offset, dep->m_req, Ghost::None, 0, fromDeviceIndex, tempGhostVar, dest);

            // let this Task GPU DW know about this staging array
            this->getTaskVars().addTaskGpuDWStagingVar(fromPatch, matlIndx, levelID, host_offset, host_size, elementDataSize, dep->m_req, fromDeviceIndex);

            // Now make sure the Task DW knows about the non-staging
            // variable where the staging variable's data will come
            // from.  Scenarios occur in which the same source region
            // is listed to send to two different patches.  This task
            // doesn't need to know about the same source twice.
            if (!(this->getTaskVars().varAlreadyExists(dep->m_req->m_var, fromPatch, matlIndx, levelID, dep->m_req->mapDataWarehouse()))) {
              // let this Task GPU DW know about the source location.
              this->getTaskVars().addTaskGpuDWVar(fromPatch, matlIndx, levelID, elementDataSize, dep->m_req, fromDeviceIndex);
            }

            // Handle a GPU-another GPU same device transfer.  We have
            // already queued up the staging array on source GPU.  Now
            // queue up the staging array on the destination GPU.
            if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
              // Indicate we want a staging array in the device.
              // TODO: We don't need a host array, it's going
              // GPU->GPU.  So get rid of tempGhostVar here.
              this->getDeviceVars().add(toPatch, matlIndx, levelID, true, host_size,
                                         tempGhostVar->getDataSize(), elementDataSize, host_offset,
                                         dep->m_req, Ghost::None, 0, toDeviceIndex, tempGhostVar, dest);

              // And the task should know of this staging array.
              this->getTaskVars().addTaskGpuDWStagingVar(toPatch, matlIndx, levelID, host_offset, host_size, elementDataSize, dep->m_req, toDeviceIndex);

            }

            // we always write this to a "foreign" staging
            // variable. We are going to copying it from the foreign =
            // false var to the foreign = true var.  Thus the patch
            // source and destination are the same, and it's staying
            // on device.
            IntVector temp(0,0,0);
            this->getGhostVars().add(dep->m_req->m_var, fromPatch, fromPatch,
                matlIndx, levelID, false, true, host_offset, host_size, dep->m_low, dep->m_high,
                OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                temp,
                fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::sameDeviceSameMpiRank);



            if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
              // GPU to GPU copies needs another entry indicating a
              // peer to peer transfer.
              this->getGhostVars().add(dep->m_req->m_var, fromPatch, toPatch,
                 matlIndx, levelID, true, true, host_offset, host_size, dep->m_low, dep->m_high,
                 OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                 dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                 temp,
                 fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                 (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::anotherDeviceSameMpiRank);

            } else if (dest == GpuUtilities::anotherMpiRank)  {
              this->getGhostVars().add(dep->m_req->m_var, fromPatch, toPatch,
                 matlIndx, levelID, true, true, host_offset, host_size, dep->m_low, dep->m_high,
                 OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                 dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                 temp,
                 fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                 (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::anotherMpiRank);

            }
          }
        }
          break;
        default: {
          std::cerr << "DetailedTask::prepareGPUDependencies(), unsupported variable type" << std::endl;
        }
      }
    }
  }
}

//______________________________________________________________________
//
// void
// DetailedTask::gpuInitialize( bool reset )
// {
//  // ARS Reset each device.
//  // Kokkos equivalent - not needed? or would there be a Kokkos init?
//  cudaError_t retVal;
//  int numDevices = 0;
//  CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&numDevices));
//  m_num_devices = numDevices;  // ARS Set but only used for information.

//  for (int i = 0; i < m_num_devices; i++) {
//    if (reset) {
//      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));
//      CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
//    }
//  }

//  // set it back to the 0th device
//  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
//  //m_current_device = 0;  // ARS Set but unused.
// }


//______________________________________________________________________
//
void DetailedTask::turnIntoASuperPatch( GPUDataWarehouse* const gpudw
                                         , const Level* const level
                                         , const IntVector& low
                                         , const IntVector& high
                                         , const VarLabel* const label
                                         , const Patch * const patch
                                         , const int matlIndx
                                         , const int levelID
                                         )
{

  // Handle superpatch stuff
  // This was originally designed for the use case of turning an entire level into a variable.
  // We need to set up the equivalent of a super patch.
  // For example, suppose a simulation has 8 patches and 2 ranks and 1 level, and this rank owns
  // patches 0, 1, 2, and 3.  Further suppose this scheduler thread is checking
  // to see the status of a patch 1 variable which has a ton of ghost cells associated
  // with it, enough to envelop all seven other patches.  Also suppose patch 1 is
  // found on the CPU, ghost cells for patches 4, 5, 6, and 7 have previously been sent to us,
  // patch 1 is needed on the GPU, and this is the first thread to process this situation.
  // This thread's job should be to claim it is responsible for processing the variable for
  // patches 0, 1, 2, and 3.  Four GPU data warehouse entries should be created, one for each
  // patch.

  // Patches 0, 1, 2, and 3 should be given the same pointer, same low, same high, (TODO: but different offsets).
  // In order to avoid concurrency problems when marking all patches in the superpatch region as
  // belonging to the superpatch, we need to avoid Dining Philosophers problem.  That is accomplished
  // by claiming patches in *sorted* order, and no scheduler thread can attempt to claim any later patch
  // if it hasn't yet claimed a former patch.  The first thread to claim all will have claimed the
  // "superpatch" region.

  // Superpatches essentially are just windows into a shared variable, it uses shared_ptrs behind the scenes
  // With this later only one alloaction or H2D transfer can be done.  This method's job is just
  // to concurrently set up all the underlying shared_ptr work.

  // Note: Superpatch approaches won't work if for some reason a prior task copied a patch in a non-superpatch
  // manner, at the current moment no known simulation will ever do this.  It is also why we try to prepare
  // the superpatch a bit upstream before concurrency checks start, and not down in prepareDeviceVars(). Brad P - 8/6/2016
  // Future note:   A lock free reference counter should also be created and set to 4 for the above example.
  // If a patch is "vacated" from the GPU, the reference counter should be reduced.  If it hits 0, it
  // shouldn't be automatically deleted, but only available for removal if the memory space hits capacity.

  bool thisThreadHandlesSuperPatchWork = false;
  char label_cstr[80];
  strcpy (label_cstr, label->getName().c_str());


  // Get all patches in the superpatch. Assuming our superpatch is the entire level.
  // This also sorts the neighbor patches by ID for us.  Note that if the current patch is
  // smaller than all the neighbors, we have to work that in too.

  Patch::selectType neighbors;
  // IntVector low, high;
  // level->computeVariableExtents(type, low, high);  // Get the low and high for the level
  level->selectPatches(low, high, neighbors);

  // mark the lowest patch as being the superpatch
  const Patch* firstPatchInSuperPatch = nullptr;
  if (neighbors.size() == 0) {
    // this must be a one patch simulation, there are no neighbors.
    firstPatchInSuperPatch = patch;
  } else {
    firstPatchInSuperPatch = neighbors[0]->getRealPatch();
    // seeing if this patch is lower in ID number than the neighbor patches.
    if (patch->getID() < firstPatchInSuperPatch->getID()) {
      firstPatchInSuperPatch = patch;
    }
  }

  // The firstPatchInSuperPatch may not have yet been handled by a prior task  (such as it being a patch
  // assigned to a different node).  So make an entry if needed.
  gpudw->putUnallocatedIfNotExists(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID,
                                                         false, make_int3(0,0,0), make_int3(0,0,0));
  thisThreadHandlesSuperPatchWork = gpudw->compareAndSwapFormASuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  // At this point the patch has been marked as a superpatch.

  if (thisThreadHandlesSuperPatchWork) {

    gpudw->setSuperPatchLowAndSize(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID,
                                   make_int3(low.x(), low.y(), low.z()),
                                   make_int3(high.x() - low.x(), high.y() - low.y(), high.z() - low.z()));

    // This thread turned the lowest ID'd patch in the region into a superpatch.  Go through *neighbor* patches
    // in the superpatch region and flag them as being a superpatch as well (the copySuperPatchInfo call below
    // can also flag it as a superpatch.
    for( unsigned int i = 0; i < neighbors.size(); i++) {
      if (neighbors[i]->getRealPatch() != firstPatchInSuperPatch) {  // This if statement is because there is no need to merge itself

        // These neighbor patches may not have yet been handled by a prior task.  So go ahead and make sure they show up in the database
        gpudw->putUnallocatedIfNotExists(label_cstr, neighbors[i]->getRealPatch()->getID(), matlIndx, levelID,
                                         false, make_int3(0,0,0), make_int3(0,0,0));

        // TODO: Ensure these variables weren't yet allocated, in use, being copied in, etc. At the time of
        // writing, this scenario didn't exist.  Some ways to solve this include 1) An "I'm using this" reference counter.
        // 2) Moving superpatch creation to the start of a timestep, and not at the start of initiateH2D, or
        // 3) predetermining at the start of a timestep what superpatch regions will be, and then we can just form
        // them together here

        // Shallow copy this neighbor patch into the superaptch
        gpudw->copySuperPatchInfo(label_cstr, firstPatchInSuperPatch->getID(), neighbors[i]->getRealPatch()->getID(), matlIndx, levelID);

      }
    }
    gpudw->compareAndSwapSetSuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  } else {
     // spin and wait until it's done.
     while (!gpudw->isSuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID));
  }
}

// ______________________________________________________________________
//
// initiateH2DCopies is a key method for the GPU Data Warehouse and the Kokkos Scheduler
// It helps manage which data needs to go H2D, what allocations and ghost cells need to be copied, etc.
// It also manages concurrency so that no two threads could process the same task.
// A general philosophy is that this section should do atomic compareAndSwaps if it find it is the one
// to allocate, copy in, or copy in with ghosts.  After any of those actions are seen to have completed
// then they can get marked as being allocated, copied in, or copied in with ghosts.

void
DetailedTask::initiateH2DCopies(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  const Task* task = this->getTask();
  this->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  // transfer some variables twice).
  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  // materials, but a single task will never run multiple levels.
  std::multimap<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        std::multimap<labelPatchMatlDependency, const Task::Dependency*>::iterator it = vars.find(lpmd);
        if (it  == vars.end() || (it != vars.end() && it->second->m_whichdw != dependantVar->m_whichdw)) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }
  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Modifies); // modifies 2nd, may require copy
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }
  for (const Task::Dependency* dependantVar = task->getComputes(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  int device_id = -1;
  // The task runs on one device.  The first patch we see can be used to tell us
  // which device we should be on.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  varIter = vars.begin();
  if (varIter != vars.end()) {
    device_id = GpuUtilities::getGpuIndexForPatch(varIter->second->getPatchesUnderDomain(this->getPatches())->get(0));
    // Base call is commented out
    // OnDemandDataWarehouse::uintahSetCudaDevice(device_id);
  }

  // Go through each unique dependent var and see if we should allocate space and/or queue it to be copied H2D.
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {

    const Task::Dependency* curDependency = varIter->second;
    const TypeDescription::Type type = curDependency->m_var->typeDescription()->getType();

    // make sure we're dealing with a variable we support
    if (type == TypeDescription::CCVariable
        || type == TypeDescription::NCVariable
        || type == TypeDescription::SFCXVariable
        || type == TypeDescription::SFCYVariable
        || type == TypeDescription::SFCZVariable
        || type == TypeDescription::PerPatch
        || type == TypeDescription::ReductionVariable) {

      constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(this->getPatches());
      constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(this->getMaterials());
      const int numPatches = patches->size();

      const int patchID = varIter->first.m_patchID;
      const Patch * patch = nullptr;
      const Level* level = nullptr;

      for (int i = 0; i < numPatches; i++) {
        if (patches->get(i)->getID() == patchID) {
          patch = patches->get(i);
          level = patch->getLevel();
        }
      }
      if (!patch) {
        printf("ERROR:\nDetailedTask::initiateD2H() patch not found.\n");
        SCI_THROW( InternalError("DetailedTask::initiateD2H() patch not found.", __FILE__, __LINE__));
      }
      const int matlID = varIter->first.m_matlIndex;
      int levelID = level->getID();
      if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
        levelID = -1;
      }

      const int deviceIndex = GpuUtilities::getGpuIndexForPatch(patch);

      // For this dependency, get its CPU Data Warehouse and GPU Datawarehouse.
      const int dwIndex = curDependency->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse* gpudw = dw->getGPUDW(deviceIndex);

      // a fix for when INF ghost cells are requested such as in RMCRT e.g. tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
      bool uses_SHRT_MAX = (curDependency->m_num_ghost_cells == SHRT_MAX);

      // Get all size information about this dependency.
      IntVector low, high; // lowOffset, highOffset;
      if (type == TypeDescription::PerPatch || type == TypeDescription::ReductionVariable) {
        low.x(0); low.y(0); low.z(0);
        high.x(0); high.y(0); high.z(0);
      } else {
        if (uses_SHRT_MAX) {
          level->computeVariableExtents(type, low, high);
        } else {
          Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
          patch->computeVariableExtents(basis, curDependency->m_var->getBoundaryLayer(), curDependency->m_gtype, curDependency->m_num_ghost_cells, low, high);
        }
      }
      const IntVector host_size = high - low;
      const size_t elementDataSize = OnDemandDataWarehouse::getTypeDescriptionSize(curDependency->m_var->typeDescription()->getSubType()->getType());
      size_t memSize = 0;
      if (type == TypeDescription::PerPatch
          || type == TypeDescription::ReductionVariable) {
        memSize = elementDataSize;
      } else {
        memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
      }

      // Set up/get status flags
      // Start by checking if an entry doesn't exist in the GPU data warehouse.  If so, create one.
      gpudw->putUnallocatedIfNotExists(curDependency->m_var->getName().c_str(), patchID, matlID, levelID, false,
                                       make_int3(low.x(), low.y(), low.z()),
                                       make_int3(host_size.x(), host_size.y(), host_size.z()));

      bool correctSize = false;
      bool allocating = false;
      bool allocated = false;
      bool copyingIn = false;
      bool validOnGPU = false;
      bool gatheringGhostCells = false;
      bool validWithGhostCellsOnGPU = false;
      bool deallocating = false;
      bool formingSuperPatch = false;
      bool superPatch = false;


      gpudw->getStatusFlagsForVariableOnGPU(correctSize, allocating, allocated, copyingIn,
                                      validOnGPU, gatheringGhostCells, validWithGhostCellsOnGPU,
                                      deallocating, formingSuperPatch, superPatch,
                                      curDependency->m_var->getName().c_str(), patchID, matlID, levelID,
                                      make_int3(low.x(), low.y(), low.z()),
                                      make_int3(host_size.x(), host_size.y(), host_size.z()));

      // correctSize allocating allocated copyingIn validOnGPU gatheringGhostCells validWithGhostCellsOnGPU deallocating formingSuperPatch superPatch
      // printf("%d %d %s %s %d %d %d: flags: %d %d %d %d %d %d %d %d %d %d\n",
      //       Uintah::Parallel::getMPIRank(), Impl::t_tid, this->getName().c_str(), curDependency->m_var->getName().c_str(), patchID, matlID, levelID,
      //       (int)correctSize, (int)allocating, (int)allocated, (int)copyingIn, (int)validOnGPU, (int)gatheringGhostCells, (int)validWithGhostCellsOnGPU,
      //       (int)deallocating, (int)formingSuperPatch, (int)superPatch
      //      );

      if (curDependency->m_dep_type == Task::Requires || curDependency->m_dep_type == Task::Modifies) {

        // For any variable, only ONE task should manage all ghost cells for it.
        // It is a giant mess to try and have two tasks simultaneously managing ghost cells for a single var.
        // So if ghost cells are required, attempt to claim that we're the ones going to manage ghost cells
        // This changes a var's status to valid awaiting ghost data if this task claims ownership of managing ghost cells
        // Otherwise the var's status is left alone (perhaps the ghost cells were already processed by another task a while ago)
        bool gatherGhostCells = false;
        if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {

          if(uses_SHRT_MAX) {
            // Turn this into a superpatch if not already done so:
            turnIntoASuperPatch(gpudw, level, low, high, curDependency->m_var, patch, matlID, levelID);

            // At the moment superpatches are gathered together through an upcoming getRegionModifiable() call.  So we
            // still need to mark it as AWAITING_GHOST_CELLS. It should trigger as one of the simpler scenarios
            // below where it knows it can gather the ghost cells host-side before sending it into GPU memory.
          }

          // See if we get to be the lucky thread that processes all ghost cells for this simulation variable
          gatherGhostCells = gpudw->compareAndSwapAwaitingGhostDataOnGPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
        }

        // commented copyingIn here to avoid a race condition between delayed copying and gathering of ghost cells
        if ( allocated && correctSize && ( /*copyingIn ||*/ validOnGPU )) {
          // This variable exists or soon will exist on the destination.  So the non-ghost cell part of this
          // variable doesn't need any more work.

          // Queue it to be added to this tasks's TaskDW.
          // It's possible this variable data already was queued to be sent in due to this patch being a ghost cell region of another patch
          // So just double check to prevent duplicates.
          if (!this->getTaskVars().varAlreadyExists(curDependency->m_var, patch, matlID, levelID, curDependency->mapDataWarehouse())) {
            this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          }

          if (gatherGhostCells) {
            // The variable's space exists or will soon exist on the GPU.  Now copy in any ghost cells
            // into the GPU and let the GPU handle the ghost cell copying logic.

            // Indicate to the scheduler later on that this variable can be marked as valid with ghost cells.
            this->getVarsToBeGhostReady().addVarToBeGhostReady(this->getName(), patch, matlID, levelID, curDependency, deviceIndex);

            std::vector<OnDemandDataWarehouse::ValidNeighbors> validNeighbors;
            dw->getValidNeighbors(curDependency->m_var, matlID, patch, curDependency->m_gtype, curDependency->m_num_ghost_cells, validNeighbors);
            for (std::vector<OnDemandDataWarehouse::ValidNeighbors>::iterator iter = validNeighbors.begin(); iter != validNeighbors.end(); ++iter) {

              const Patch* sourcePatch = nullptr;
              if (iter->neighborPatch->getID() >= 0) {
                sourcePatch = iter->neighborPatch;
              } else {
                // This occurs on virtual patches.  They can be "wrap around" patches, meaning if you go to one end of a domain
                // you will show up on the other side.  Virtual patches have negative patch IDs, but they know what real patch they
                // are referring to.
                sourcePatch = iter->neighborPatch->getRealPatch();
              }

              IntVector ghost_host_low(0,0,0), ghost_host_high(0,0,0), ghost_host_size(0,0,0);
              IntVector ghost_host_offset(0,0,0), ghost_host_strides(0,0,0);

              IntVector virtualOffset = iter->neighborPatch->getVirtualOffset();

              int sourceDeviceNum = GpuUtilities::getGpuIndexForPatch(sourcePatch);
              int destDeviceNum = deviceIndex;

              // Find out who has our ghost cells.  Listed in priority...
              // It could be in the GPU as a staging/foreign var
              // Or in the GPU as a full variable
              // Or in the CPU as a foreign var
              // Or in the CPU as a regular variable
              bool useGpuStaging = false;
              bool useGpuGhostCells = false;
              bool useCpuForeign = false;
              bool useCpuGhostCells = false;

              // See if it's in the GPU as a staging/foreign var
              useGpuStaging = gpudw->stagingVarExists(curDependency->m_var->getName().c_str(),
                                                    patchID, matlID, levelID,
                                                    make_int3(iter->low.x(), iter->low.y(), iter->low.z()),
                                                    make_int3(iter->high.x() - iter->low.x(), iter->high.y()- iter->low.y(), iter->high.z()- iter->low.z()));

              // See if we have the entire neighbor patch in the GPU (not just a staging)
              useGpuGhostCells = gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(), sourcePatch->getID(), matlID, levelID);

              // See if we have CPU foreign var data or just the plain CPU variable we can use
              // Note: We don't have a full system in place to set valid all CPU variables.  Specifically foreign variables are not set, and
              // so the line below is commented out.  In the meantime assume that if it's not on the GPU, it must be on the CPU.
              // if (gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(), sourcePatch->getID(), matlID, levelID)) {
                if (iter->validNeighbor && iter->validNeighbor->isForeign()) {
                  useCpuForeign = true;
                } else {
                  useCpuGhostCells = true;
                }
              // }


              // get the sizes of the source variable
              if (useGpuStaging) {
                ghost_host_low = iter->low;
                ghost_host_high = iter->high;
                ghost_host_size = ghost_host_high - ghost_host_low;
              } else if (useGpuGhostCells) {
                GPUDataWarehouse::GhostType throwaway1;
                int throwaway2;
                int3 ghost_host_low3, ghost_host_high3, ghost_host_size3;

                gpudw->getSizes(ghost_host_low3, ghost_host_high3, ghost_host_size3, throwaway1, throwaway2,
                               curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
                ghost_host_low = IntVector(ghost_host_low3.x, ghost_host_low3.y, ghost_host_low3.z);
                ghost_host_high = IntVector(ghost_host_high3.x, ghost_host_high3.y, ghost_host_high3.z);
                ghost_host_size = IntVector(ghost_host_size3.x, ghost_host_size3.y, ghost_host_size3.z);

              } else if (useCpuForeign || useCpuGhostCells) {
                iter->validNeighbor->getSizes(ghost_host_low, ghost_host_high, ghost_host_offset, ghost_host_size, ghost_host_strides);
              }
              const size_t ghost_mem_size =  ghost_host_size.x() * ghost_host_size.y() * ghost_host_size.z() * elementDataSize;

              if (useGpuStaging) {

                // Make sure this task GPU DW knows about the staging var
                this->getTaskVars().addTaskGpuDWStagingVar(patch, matlID, levelID, iter->low, iter->high - iter->low,
                                                            elementDataSize, curDependency, destDeviceNum);

                // Assume for now that the ghost cell region is also the exact same size as the
                // staging var.  (If in the future ghost cell data is managed a bit better as
                // it currently does on the CPU, then some ghost cell regions will be found
                // *within* an existing staging var.  This is known to happen with Wasatch
                // computations involving periodic boundary scenarios.)
                this->getGhostVars().add(curDependency->m_var,
                    patch, patch,   /*We're merging the staging variable on in*/
                    matlID, levelID,
                    true, false,
                    iter->low,              /*Assuming ghost cell region is the variable size */
                    IntVector(iter->high.x() - iter->low.x(), iter->high.y() - iter->low.y(), iter->high.z() - iter->low.z()),
                    iter->low,
                    iter->high,
                    elementDataSize,
                    curDependency->m_var->typeDescription()->getSubType()->getType(),
                    virtualOffset,
                    destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                    (Task::WhichDW) curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);

              } else if (useGpuGhostCells) {

                // If this task doesn't own this source patch, then we need to make sure
                // the upcoming task data warehouse at least has knowledge of this GPU variable that
                // already exists in the GPU.  So queue up to load the neighbor patch metadata into the
                // task datawarehouse.
                if (!patches->contains(sourcePatch)) {
                  if (!(this->getTaskVars().varAlreadyExists(curDependency->m_var, sourcePatch, matlID, levelID,
                                                                (Task::WhichDW) curDependency->mapDataWarehouse()))) {
                      this->getTaskVars().addTaskGpuDWVar(sourcePatch, matlID, levelID,
                                                           elementDataSize, curDependency, sourceDeviceNum);
                  }
                }

                // Store the source and destination patch, and the range of the ghost cells
                // A GPU kernel will use this collection to do all internal GPU ghost cell copies for
                // that one specific GPU.
                this->getGhostVars().add(curDependency->m_var,
                    sourcePatch, patch, matlID, levelID,
                    false, false,
                    ghost_host_low, ghost_host_size,
                    iter->low, iter->high,
                    elementDataSize,
                    curDependency->m_var->typeDescription()->getSubType()->getType(),
                    virtualOffset,
                    destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                    (Task::WhichDW) curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);

              } else if (useCpuForeign) {

                // Prepare to tell the host-side GPU DW to allocate space for this variable.
                // Since we already got the gridVariableBase pointer to that foreign var, go ahead and add it in here.
                // (The OnDemandDataWarehouse is weird, it doesn't let you query foreign vars, it will try to inflate a regular
                // var and deep copy the foreign var on in.  So for now, just pass in the pointer.)
                this->getDeviceVars().add(sourcePatch, matlID, levelID, true, ghost_host_size, ghost_mem_size,
                                           elementDataSize, ghost_host_low, curDependency, Ghost::None, 0,
                                           destDeviceNum, iter->validNeighbor, GpuUtilities::sameDeviceSameMpiRank);

                // Let this Task GPU DW know about this staging array.  We may end up not needed it if another thread processes it or it became
                // part of a superpatch.  We'll figure that out later when we go actually add it.
                this->getTaskVars().addTaskGpuDWStagingVar(sourcePatch, matlID, levelID, ghost_host_low, ghost_host_size,
                                                            elementDataSize, curDependency, sourceDeviceNum);

                this->getGhostVars().add(curDependency->m_var, sourcePatch, patch, matlID, levelID,
                                          true, false, ghost_host_low, ghost_host_size,
                                          iter->low, iter->high, elementDataSize,
                                          curDependency->m_var->typeDescription()->getSubType()->getType(), virtualOffset,
                                          destDeviceNum, destDeviceNum, -1, -1, /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                                          (Task::WhichDW)curDependency->mapDataWarehouse(),
                                          GpuUtilities::sameDeviceSameMpiRank);

              } else if (useCpuGhostCells) {
                // This handles the scenario where the variable is in the GPU, but the ghost cell data is only found in the
                // neighboring normal patch (non-foreign) in host memory.  Ghost cells haven't been gathered in or started
                // to be gathered in.

                // Check if we should copy this patch into the GPU.

                // TODO: Instead of copying the entire patch for a ghost cell, we should just create a foreign var, copy
                // a contiguous array of ghost cell data into that foreign var, then copy in that foreign var.  If it's a foreign var,
                // then the foreign var section above should handle it, not here.
                if (!this->getDeviceVars().varAlreadyExists(curDependency->m_var, sourcePatch, matlID, levelID, curDependency->mapDataWarehouse())) {

                  // Prepare to tell the host-side GPU DW to possibly allocate and/or copy this variable.
                  this->getDeviceVars().add(sourcePatch, matlID, levelID, false,
                      ghost_host_size, ghost_mem_size,
                      elementDataSize, ghost_host_low,
                      curDependency, Ghost::None, 0, destDeviceNum,
                      nullptr, GpuUtilities::sameDeviceSameMpiRank);

                  // Prepare this task GPU DW for knowing about this variable on the GPU.
                  this->getTaskVars().addTaskGpuDWVar(sourcePatch, matlID, levelID, elementDataSize, curDependency, destDeviceNum);

                } // else the variable is already in deviceVars

                // Add in info to perform a GPU ghost cell copy.  (It will ensure duplicates can't be entered.)
                this->getGhostVars().add(curDependency->m_var,
                                          sourcePatch, patch, matlID, levelID,
                                          false, false,
                                          ghost_host_low, ghost_host_size,
                                          iter->low, iter->high,
                                          elementDataSize,
                                          curDependency->m_var->typeDescription()->getSubType()->getType(),
                                          virtualOffset,
                                          destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                                          (Task::WhichDW) curDependency->mapDataWarehouse(),
                                          GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("%s ERROR: Needed ghost cell data not found on the CPU or a GPU.  Looking for ghost cells to be sent to label %s patch %d matl %d.  Couldn't find the source from patch %d.\n",
                    myRankThread().c_str(), curDependency->m_var->getName().c_str(), patchID, matlID, sourcePatch->getID());
                SCI_THROW(InternalError("Needed ghost cell data not found on the CPU or a GPU\n",__FILE__, __LINE__));
              }
            } // end neighbors for loop
          } // end if(gatherGhostCells)
        } else if ( allocated && !correctSize ) {
          // At the moment this isn't allowed.  There are two reasons for this.
          // First, the current CPU system always needs to "resize" variables when ghost cells are required.
          // Essentially the variables weren't created with room for ghost cells, and so room  needs to be created.
          // This step can be somewhat costly (I've seen a benchmark where it took 5% of the total computation time).
          // And at the moment this hasn't been coded to resize on the GPU.  It would require an additional step and
          // synchronization to make it work.
          // The second reason is with concurrency.  Suppose a patch that CPU thread A own needs
          // ghost cells from a patch that CPU thread B owns.
          // A can recognize that B's data is valid on the GPU, and so it stores for the future to copy B's
          // data on in.  Meanwhile B notices it needs to resize.  So A could start trying to copy in B's
          // ghost cell data while B is resizing its own data.
          // I believe both issues can be fixed with proper checkpoints.  But in reality
          // we shouldn't be resizing variables on the GPU, so this event should never happen.

          // DS 04222020: AMRSimulationController::collectGhostCells and SchedulerCommon::addTask together can
          // determine max ghost cells for variables across all tasks and across init and main task graph.
          // So this scenario can only occur when there are other task graphs such as sub scheduler of Poisson2
          // or may be during AMR as methods related to it are not yet included in collectGhostCells.
          // Updating error message to more meaningful text.
          gpudw->remove(curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
          std::cout <<  myRankThread()
                    // << " Resizing of GPU grid vars not implemented at this time. "
                    <<"\n**Ensure the MAX number of ghost cells for the variable for GPU tasks in the previous task graph are same as in the current taskgraph**\n "
                    << "Task: " << this->getName()
                    // << "For the GPU, computes need to be declared with scratch computes to have room for ghost cells.  "
                    << " for " << curDependency->m_var->getName()
                    << " patch " << patchID
                    << " material " << matlID
                    << " level " << levelID
                    << ".  Requested var of size (" << host_size.x() << ", " << host_size.y() << ", " << host_size.z() << ") "
                    << "with offset (" << low.x() << ", " << low.y() << ", " << low.z() << ")"
                    << " max ghost cells set to: " << curDependency->m_var->getMaxDeviceGhost()
                    << "\n Are you using sub scheduler or AMR? Those are not yet supported by AMRSimulationController::collectGhostCells."
                    << std::endl;

          SCI_THROW(InternalError("ERROR: Resizing of GPU grid vars not implemented at this time",__FILE__, __LINE__));

        // commented copyingIn here to avoid a race condition between delayed copying and gathering of ghost cells
        } else if (( !allocated )
                   || ( allocated && correctSize && !validOnGPU /*&& !copyingIn*/ )) {

          // It's either not on the GPU, or space exists on the GPU for it but it is invalid.
          // Either way, gather all ghost cells host side (if needed), then queue the data to be
          // copied in H2D.  If the data doesn't exist in the GPU, then the upcoming allocateAndPut
          // will allocate space for it.  Otherwise if it does exist on the GPU, the upcoming
          // allocateAndPut will notice that and simply configure it to reuse the pointer.

          if (type == TypeDescription::CCVariable
              || type == TypeDescription::NCVariable
              || type == TypeDescription::SFCXVariable
              || type == TypeDescription::SFCYVariable
              || type == TypeDescription::SFCZVariable) {

            // Queue this CPU var to go into the host-side GPU DW.
            // Also queue that this GPU DW var should also be found in this tasks's Task DW.

            this->getDeviceVars().add(patch, matlID, levelID, false, host_size, memSize, elementDataSize,
                                       low, curDependency, curDependency->m_gtype, curDependency->m_num_ghost_cells, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);

            // Mark that when this variable is copied in, it will have its ghost cells ready too.
            if (gatherGhostCells) {
              this->getVarsToBeGhostReady().addVarToBeGhostReady(this->getName(), patch, matlID, levelID, curDependency, deviceIndex);
            }
          } else if (type == TypeDescription::PerPatch) {
            // PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(curDependency->m_var->typeDescription()->createInstance());
            // dw->get(*patchVar, curDependency->m_var, matlID, patch);
            this->getDeviceVars().add(patch, matlID, levelID, elementDataSize, elementDataSize, curDependency, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          } else if (type == TypeDescription::ReductionVariable) {
            levelID = -1;
            // ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(curDependency->m_var->typeDescription()->createInstance());
            // dw->get(*reductionVar, curDependency->m_var, patch->getLevel(), matlID);
            this->getDeviceVars().add(patch, matlID, levelID, elementDataSize, elementDataSize, curDependency, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          }
          else {
            std::cerr << "DetailedTask::initiateH2D(), unsupported variable type for computes variable "
                      << curDependency->m_var->getName() << std::endl;
          }
        }
      } else if (curDependency->m_dep_type == Task::Computes) {
        // compute the amount of space the host needs to reserve on the GPU for this variable.

        if (type == TypeDescription::PerPatch) {
          // For PerPatch, it's not a mesh of variables, it's just a single variable, so elementDataSize is the memSize.
          this->getDeviceVars().add(patch, matlID, levelID, memSize, elementDataSize, curDependency, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
        } else if (type == TypeDescription::ReductionVariable) {
          // For ReductionVariable, it's not a mesh of variables, it's just a single variable, so elementDataSize is the memSize.
          this->getDeviceVars().add(patch, matlID, levelID, memSize, elementDataSize, curDependency, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);

        } else if (type == TypeDescription::CCVariable
            || type == TypeDescription::NCVariable
            || type == TypeDescription::SFCXVariable
            || type == TypeDescription::SFCYVariable
            || type == TypeDescription::SFCZVariable) {

          this->getDeviceVars().add(patch, matlID, levelID, false, host_size, memSize, elementDataSize, low, curDependency,
                                     curDependency->m_gtype, curDependency->m_num_ghost_cells, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          this->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
        } else {
          std::cerr << "DetailedTask::initiateH2D(), unsupported variable type for computes variable "
                    << curDependency->m_var->getName() << std::endl;
        }
      }
    }
  }

  // We've now gathered up all possible things that need to go on the device.  Copy it over.

  createTaskGpuDWs();

  prepareDeviceVars(m_dws);

  // At this point all needed variables will have a pointer.

  prepareTaskVarsIntoTaskDW(m_dws);

  prepareGhostCellsIntoTaskDW();

}


// ______________________________________________________________________
//
void
DetailedTask::prepareDeviceVars(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  bool isStaging = false;

  std::string taskID = this->getName();
  // for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
  isStaging = false;
  // Because maps are unordered, it is possible a staging var could be inserted before the regular var exists.
  // So just loop twice, once when all staging is false, then loop again when all staging is true
  for (int i = 0; i < 2; i++) {
    // Get all data in the GPU, and store it on the GPU Data Warehouse on the host, as only it
    // is responsible for management of data.  So this processes the previously collected deviceVars.
    std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = this->getDeviceVars().getMap();

    for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
        it != varMap.end(); ++it) {
      int whichGPU = it->second.m_whichGPU;
      int dwIndex = it->second.m_dep->mapDataWarehouse();

      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse* gpudw = dw->getGPUDW(whichGPU);
      if (!gpudw) {
        SCI_THROW(InternalError("No GPU data warehouse found\n",__FILE__, __LINE__));
      }

      if (it->second.m_staging == isStaging) {

        if (this->getDeviceVars().getTotalVars(whichGPU, dwIndex)) {

          void* device_ptr = nullptr;  // device base pointer to raw data

          const IntVector offset = it->second.m_offset;
          const IntVector size = it->second.m_sizeVector;
          IntVector low = offset;           // DS 12132019: GPU Resize fix. update low and high to max ghost cell if needed
          IntVector high = offset + size;
          const TypeDescription* type_description = it->second.m_dep->m_var->typeDescription();
          const TypeDescription::Type type = type_description->getType();
          const TypeDescription::Type subtype = type_description->getSubType()->getType();
          const VarLabel* label = it->second.m_dep->m_var;
          char label_cstr[80];
          strcpy (label_cstr, it->second.m_dep->m_var->getName().c_str());
          const Patch* patch = it->second.m_patchPointer;
          const int patchID = it->first.m_patchID;
          const int matlIndx = it->first.m_matlIndx;
          const int levelID = it->first.m_levelIndx;
          const size_t elementDataSize = it->second.m_sizeOfDataType;
          const bool staging = it->second.m_staging;
          const int numGhostCells = it->second.m_numGhostCells;
          Ghost::GhostType ghosttype = it->second.m_gtype;
          bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);

          // DS 12132019: GPU Resize fix. getting max ghost cell related info
          const IntVector boundaryLayer = it->second.m_dep->m_var->getBoundaryLayer();
          Ghost::GhostType dgtype = it->second.m_dep->m_var->getMaxDeviceGhostType();
          int dghost = it->second.m_dep->m_var->getMaxDeviceGhost();

          if(numGhostCells>dghost){// RMCRT
            dghost = numGhostCells;
            dgtype = ghosttype;
          }

          // Allocate the vars if needed.  If they've already been allocated, then
          // this simply sets the var to reuse the existing pointer.
          switch (type) {
            case TypeDescription::PerPatch : {
              GPUPerPatchBase* patchVar = OnDemandDataWarehouse::createGPUPerPatch(subtype);
              gpudw->allocateAndPut(*patchVar, label_cstr, patchID, matlIndx, levelID, elementDataSize);
              device_ptr = patchVar->getVoidPointer();
              delete patchVar;
              break;
            }
            case TypeDescription::ReductionVariable : {

              GPUReductionVariableBase* reductionVar = OnDemandDataWarehouse::createGPUReductionVariable(subtype);
              gpudw->allocateAndPut(*reductionVar, label_cstr, patchID, matlIndx, levelID, elementDataSize);
              device_ptr = reductionVar->getVoidPointer();
              delete reductionVar;
              break;
            }
            case TypeDescription::CCVariable :
            case TypeDescription::NCVariable :
            case TypeDescription::SFCXVariable :
            case TypeDescription::SFCYVariable :
            case TypeDescription::SFCZVariable : {
              GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(subtype);

              if (!uses_SHRT_MAX) {
                // DS 12132019: GPU Resize fix. Do it only if its not staging. Use max ghost cells and corresponding low and high to allocate scratch space
                if (it->second.m_staging==false) {
                  const TypeDescription::Type type = it->second.m_dep->m_var->typeDescription()->getType();
                  Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
                  patch->computeVariableExtents(basis, it->second.m_dep->m_var->getBoundaryLayer(), dgtype, dghost, low, high);

                  // DS 01022019: set CPU status to valid for the variable to be copied. Otherwise picks up default status of Unknown on host and
                  // copies it back for the next requires CPU.
                  // if (it->second.m_dep->m_dep_type == Task::Requires) {
                  //  gpudw->compareAndSwapSetValidOnCPU(label_cstr, patchID, matlIndx, levelID);
                  // }
                }
                gpudw->allocateAndPut(*device_var, label_cstr, patchID, matlIndx, levelID, staging,
                                      make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()),
                                      elementDataSize, (GPUDataWarehouse::GhostType)(dgtype),
                                      dghost);
              } else {

                // TODO, give it an offset so it could be requested as a patch or as a level.  Right now they all get the same low/high.
                gpudw->allocateAndPut(*device_var, label_cstr, patchID, matlIndx, levelID, staging,
                                        make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()),
                                        elementDataSize, (GPUDataWarehouse::GhostType)(it->second.m_gtype),
                                        it->second.m_numGhostCells);

              }
              device_ptr = device_var->getVoidPointer();
              delete device_var;
              break;
            }
            default : {
              cerrLock.lock();
              {
                std::cerr << "This variable's type is not supported." << std::endl;
              }
              cerrLock.unlock();
            }
          }

          // If it's a requires, copy the data on over.  If it's a
          // computes, leave it as allocated but unused space.
          if (it->second.m_dep->m_dep_type == Task::Requires || it->second.m_dep->m_dep_type == Task::Modifies) {
            if (!device_ptr) {
              std::cout << myRankThread() << " ERROR: GPU variable's device pointer was nullptr. "
                  << "For " << label_cstr
                  << " patch " << patchID
                  << " material " << matlIndx
                  << " level " << levelID << "." << std::endl;
              SCI_THROW(InternalError("ERROR: GPU variable's device pointer was nullptr",__FILE__, __LINE__));
            }

            if (it->second.m_dest == GpuUtilities::sameDeviceSameMpiRank) {

              // Figure out which thread gets to copy data H2D.  First
              // touch wins.  In case of a superpatch, the patch vars
              // were shallow copied so they all patches in the
              // superpatch refer to the same atomic status.
              bool performCopy = false;
              bool delayedCopy = false;
              if (!staging) {
                performCopy = gpudw->compareAndSwapCopyingIntoGPU(label_cstr, patchID, matlIndx, levelID, numGhostCells);
              }
              else {
                performCopy = gpudw->compareAndSwapCopyingIntoGPUStaging(label_cstr, patchID, matlIndx, levelID,
                                                                     make_int3(low.x(), low.y(), low.z()),
                                                                     make_int3(size.x(), size.y(), size.z()));
              }

              if (performCopy == false && !staging && numGhostCells > 0) {
                // delayedCopy: call if gpudw->delayedCopyNeeded. It
                // is needed if current patch on GPU (either valid or
                // being copied) has lesser number of ghost cells than
                // current) delayed copy is needed to avoid race
                // conditions and avoid the current copy with larger
                // ghost layer being overwritten by those queued
                // earlier with smaller ghost layer. This race
                // condition was observed due to using different
                // streams.
                delayedCopy = gpudw->isDelayedCopyingNeededOnGPU(label_cstr, patchID, matlIndx, levelID, numGhostCells);
              }


              if (performCopy || delayedCopy) { // delayedCopy: use
                                                // performCopy ||
                                                // delayedCopy

                // This thread is doing the H2D copy for this
                // simulation variable.

                // Start by getting the host pointer.
                void* host_ptr = nullptr;

                // The variable exists in host memory.  We just have
                // to get one and copy it on in.
                switch (type) {
                  case TypeDescription::CCVariable :
                  case TypeDescription::NCVariable :
                  case TypeDescription::SFCXVariable :
                  case TypeDescription::SFCYVariable :
                  case TypeDescription::SFCZVariable : {

                    // The var on the host could either be a regular
                    // var or a foreign var.  If it's a regular var,
                    // this will manage ghost cells by creating a host
                    // var, rewindowing it, then copying in the
                    // regions needed for the ghost cells.  If this is
                    // the case, then ghost cells for this specific
                    // instance of this var is completed.  If it's a
                    // foreign var, then there is no API at the moment
                    // to query it directly (if you try to getGridVar
                    // a foreign var, it doesn't work, it wasn't
                    // designed for that).  Fortunately we would have
                    // already seen it in whatever function called
                    // this. So use that instead.

                    // Note: Unhandled scenario: If the adjacent patch
                    // is only in the GPU, this code doesn't gather
                    // it.
                    if (uses_SHRT_MAX) {
                      g_GridVarSuperPatch_mutex.lock();
                      {

                        // The variable wants the entire domain.  So we
                        // do a getRegion call instead.
                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(type_description->createInstance());

                        // dw->allocateAndPut(*gridVar, label, matlIndx, patch, ghosttype, numGhostCells, true);
                        if (!dw->exists(label, matlIndx, patch->getLevel())) {
                          // This creates and deep copies a region from
                          // the OnDemandDatawarehouse.  It does so by
                          // deep copying from the other patches and
                          // forming one large region.
                          dw ->getRegionModifiable(*gridVar, label, matlIndx, patch->getLevel(), low, high, true);
                          // Passing in a clone (really it's just a
                          // shallow copy) to increase the reference
                          // counter by one.
                          dw->putLevelDB(gridVar->clone(), label, patch->getLevel(), matlIndx);
                          // dw->getLevel(*constGridVar, label, matlIndx, patch->getLevel());
                        } else {
                          exit(-1);
                        }
                        // Get the host pointer as well
                        host_ptr = gridVar->getBasePointer();

                        // Let go of the reference, allowing a single
                        // reference to remain and keep the variable
                        // alive in leveDB.  delete gridVar;

                        // TODO: Verify this cleans up.  If so change the comment.
                      }
                      g_GridVarSuperPatch_mutex.unlock();
                    } else {
                      if (it->second.m_var)  {
                        // It's a foreign var.  We can't look it up,
                        // but we saw it previously.
                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(it->second.m_var);
                        host_ptr = gridVar->getBasePointer();
                        // Since we didn't do a getGridVar() call, no
                        // reference to clean up.
                      } else {
                        // I'm commenting carefully because this section has bit me several times.  If it's not done right, the bugs
                        // are a major headache to track down.  -- Brad P. Nov 30, 2016
                        // We need all the data in the patch.  Perform a getGridVar(), which will return a var with the same window/data as the
                        // OnDemand DW variable, or it will create a new window/data sized to hold the room of the ghost cells and copy it into
                        // the gridVar variable.  Internally it keeps track of refcounts for the window object and the data object.
                        // In any scenario treat the gridVar as a temporary copy of the actual var in the OnDemand DW,
                        // and as such that temporary variable needs to be reclaimed so there are no memory leaks.  The problem is that
                        // we need this temporary variable to live long enough to perform a device-to-host copy.
                        // * In one scenario with no ghost cells, you get back the same window/data just with refcounts incremented by 1.
                        // * In another scenario with ghost cells, the ref counts are at least 2, so deleting the gridVar won't automatically deallocate it
                        // * In another scenario with ghost cells, you get back a gridvar holding different window/data, their refcounts are 1
                        //  and so so deleting the gridVar will invoke deallocation.  That would be bad if an async device-to-host copy is needed.
                        // In all scenarios, the correct approach is just to delay deleting the gridVar object, and letting it persist until the
                        // all variable copies complete, then delete the object, which in turn decrements the refcounter, which then allows it to clean
                        // up later where needed (either immediately if the temp's refcounts hit 0, or later when the it does the scrub checks).

                        // DS 12132019: GPU Resize fix. Do it only if its not staging. Use max ghost cells and corresponding low and high to allocate scratch space
                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(type_description->createInstance());
                        dw->getGridVar(*gridVar, label, matlIndx, patch, dgtype, numGhostCells);
                        host_ptr = gridVar->getBasePointer();
                        it->second.m_tempVarToReclaim = gridVar;  // This will be held onto so it persists, and then cleaned up after the device-to-host copy
                        if (it->second.m_staging == false) {
                          it->second.m_varMemSize = gridVar->getDataSize(); // update it->second.m_varMemSize to add scratchGhost;
                        }
                      }
                    }
                    break;
                  }
                  case TypeDescription::PerPatch : {
                    PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(type_description->createInstance());
                    dw->get(*patchVar, label, matlIndx, patch);
                    host_ptr = patchVar->getBasePointer();
                    // let go of our reference
                    delete patchVar;
                    break;
                  }
                  case TypeDescription::ReductionVariable : {
                    ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(type_description->createInstance());
                    dw->get(*reductionVar, label, patch->getLevel(), matlIndx);
                    host_ptr = reductionVar->getBasePointer();
                    // let go of our reference
                    delete reductionVar;
                    break;
                  }
                  default : {
                    cerrLock.lock();
                    {
                      std::cerr << "Variable " << label_cstr
                                << " is of a type that is not supported on GPUs yet."
                                << std::endl;
                    }
                    cerrLock.unlock();
                  }
                }

                if (host_ptr && device_ptr) {
                  // Perform the copy!

                  // Base call is commented out
                  // OnDemandDataWarehouse::uintahSetCudaDevice(whichGPU);

                  if (it->second.m_varMemSize == 0) {
                    printf("ERROR: For variable %s patch %d material %d level %d staging %s attempting to copy zero bytes to the GPU.\n",
                        label_cstr, patchID, matlIndx, levelID, staging ? "true" : "false" );
                    SCI_THROW(InternalError("Attempting to copy zero bytes to the GPU.  That shouldn't happen.", __FILE__, __LINE__));
                  }

                  // Debug loop in case you need to see the data being sent.
                  // if (it->second.m_varMemSize == 968) {
                  //  printf("DetailedTask - d_data is %p\n", host_ptr);
                  //  for (int i = 0; i < 968/elementDataSize; i++) {
                  //    printf("DetailedTask - Array at index %d is %1.6lf\n", i, *(static_cast<double*>(host_ptr) + i));
                  //  }
                  // }

                  if (delayedCopy) {
//                      printf("%s task: %s, prepareDeviceVars - mark for delayed copying %s %d %d %d %d: %d\n",myRankThread().c_str(), this->getName().c_str(), it->first.m_label.c_str(),
//                              it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx, it->first.m_dataWarehouse, numGhostCells);
                    this->getDelayedCopyingVars().push_back(DetailedTask::delayedCopyingInfo(it->first, it->second, device_ptr, host_ptr, it->second.m_varMemSize));
//                    while(gpudw->isValidOnGPU(label_cstr, patchID, matlIndx, levelID)==false);
                  } else {
//                      printf("%s task: %s, prepareDeviceVars - copying %s %d %d %d %d: %d\n",myRankThread().c_str(), this->getName().c_str(), it->first.m_label.c_str(),
//                                                    it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx, it->first.m_dataWarehouse, numGhostCells);
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
                    m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
                    m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                              whichGPU,
                                              device_ptr, host_ptr,
                                              it->second.m_varMemSize, cudaMemcpyHostToDevice);
#else
                    cudaStream_t* stream = this->getCudaStreamForThisTask(whichGPU);
                    // ARS cudaMemcpyAsync copy from host to device
                    // Kokkos equivalent - deep copy.
                    CUDA_RT_SAFE_CALL(cudaMemcpyAsync(device_ptr, host_ptr, it->second.m_varMemSize, cudaMemcpyHostToDevice, *stream));
#endif
                    // Tell this task that we're managing the copies for this variable.

                    this->getVarsBeingCopiedByTask().getMap().insert(std::pair<GpuUtilities::LabelPatchMatlLevelDw,
                                                                DeviceGridVariableInfo>(it->first, it->second));
                  }
                }
              }
            } else if (it->second.m_dest == GpuUtilities::anotherDeviceSameMpiRank ||
                       it->second.m_dest == GpuUtilities::anotherMpiRank) {
              // We're not performing a host to GPU copy.  This is
              // just prepare a staging var.  So it is a a gpu normal
              // var to gpu staging var copy.  It is to prepare for
              // upcoming GPU to host (MPI) or GPU to GPU copies.
              // Tell this task that we're managing the copies for
              // this variable.
              this->getVarsBeingCopiedByTask().getMap().insert(
                      std::pair<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>(it->first, it->second));
            }
          }
        }
      }
    }
    isStaging = !isStaging;
  }
  // This code is commented out for now until multi-device support is added.
  // } end for (deviceNumSetIter deviceNums_it = m_deviceNums.begin()
}


// ______________________________________________________________________
//
void
DetailedTask::copyDelayedDeviceVars()
{
  this->getVarsBeingCopiedByTask().getMap().clear();

  for (auto it = this->getDelayedCopyingVars().begin(); it != this->getDelayedCopyingVars().end(); it++) {
    GpuUtilities::LabelPatchMatlLevelDw lpmld = it->lpmld;
    DeviceGridVariableInfo devGridVarInfo = it->devGridVarInfo;

    void * device_ptr = it->device_ptr;
    void * host_ptr = it->host_ptr;

    size_t size = it->size;

//    printf("%s task: %s, delayed copying %s %d %d %d %d\n",myRankThread().c_str(), this->getName().c_str(), lpmld.m_label.c_str(),
//          lpmld.m_patchID, lpmld.m_matlIndx, lpmld.m_levelIndx, lpmld.m_dataWarehouse);

#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
    m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
    m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                              devGridVarInfo.m_whichGPU,
                              device_ptr, host_ptr,
                              size, cudaMemcpyHostToDevice);
#else
    cudaStream_t* stream = this->getCudaStreamForThisTask(devGridVarInfo.m_whichGPU);
    // ARS cudaMemcpyAsync copy from host to device
    // Kokkos equivalent - deep copy.
    CUDA_RT_SAFE_CALL(cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, *stream));
#endif

    // Tell this task that we're managing the copies for this variable.
    this->getVarsBeingCopiedByTask().getMap().insert(std::pair<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>(lpmld, devGridVarInfo));
  }

  this->getDelayedCopyingVars().clear();
}


// ______________________________________________________________________
//
bool
DetailedTask::delayedDeviceVarsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  for (auto it = this->getDelayedCopyingVars().begin(); it != this->getDelayedCopyingVars().end(); it++) {
    GpuUtilities::LabelPatchMatlLevelDw lpmld = it->lpmld;
    DeviceGridVariableInfo devGridVarInfo = it->devGridVarInfo;

    int whichGPU = devGridVarInfo.m_whichGPU;
    int dwIndex = devGridVarInfo.m_dep->mapDataWarehouse();

    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse* gpudw = dw->getGPUDW(whichGPU);

    if (!(gpudw->isValidOnGPU(lpmld.m_label.c_str(), lpmld.m_patchID, lpmld.m_matlIndx, lpmld.m_levelIndx ))) {
      return false;
    }
  }

  return true;
}


// ______________________________________________________________________
//
void
DetailedTask::prepareTaskVarsIntoTaskDW(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Copy all task variables metadata into the Task GPU DW.  All
  // necessary metadata information must already exist in the
  // host-side GPU DWs.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & taskVarMap = this->getTaskVars().getMap();

  // Because maps are unordered, it is possible a staging var could be
  // inserted before the regular var exists.  So just loop twice, once
  // when all staging is false, then loop again when all staging is true
  bool isStaging = false;

  for (int i = 0; i < 2; i++) {
    std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator it;
    for (it = taskVarMap.begin(); it != taskVarMap.end(); ++it) {
      // If isStaging is false, do the non-staging vars, then if
      // isStaging is true, do the staging vars.  isStaging is flipped
      // after the first iteration of the i for loop.
      if (it->second.m_staging == isStaging) {
        switch (it->second.m_dep->m_var->typeDescription()->getType()) {
          case TypeDescription::PerPatch :
          case TypeDescription::ReductionVariable :
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            int dwIndex = it->second.m_dep->mapDataWarehouse();
            GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(it->second.m_whichGPU);
            int patchID = it->first.m_patchID;
            int matlIndx = it->first.m_matlIndx;
            int levelIndx = it->first.m_levelIndx;

            int3 offset;
            int3 size;
            if (it->second.m_staging) {
              offset = make_int3(it->second.m_offset.x(), it->second.m_offset.y(), it->second.m_offset.z());
              size = make_int3(it->second.m_sizeVector.x(), it->second.m_sizeVector.y(), it->second.m_sizeVector.z());
            }
            else {
              offset = make_int3(0, 0, 0);
              size = make_int3(0, 0, 0);
            }

            GPUDataWarehouse* taskgpudw =
              this->getTaskGpuDataWarehouse(it->second.m_whichGPU,
                                            (Task::WhichDW) dwIndex);
            if (taskgpudw) {
              taskgpudw->copyItemIntoTaskDW(gpudw,
                                            it->second.m_dep->m_var->getName().c_str(),
                                            patchID, matlIndx, levelIndx,
                                            it->second.m_staging, offset, size);
            }
            else {
              printf("ERROR - No task data warehouse found for device %d "
                     "for task %s\n",
                     it->second.m_whichGPU,
                     this->getTask()->getName().c_str());
              SCI_THROW(InternalError("No task data warehouse found\n",
                                      __FILE__, __LINE__));
            }
          }
            break;
          default : {
            cerrLock.lock();
            {
              std::cerr << "Variable " << it->second.m_dep->m_var->getName()
                        << " is of a type that is not supported on GPUs yet."
                        << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
    isStaging = !isStaging;
  }
}


// ______________________________________________________________________
//
void
DetailedTask::prepareGhostCellsIntoTaskDW()
{
  // Tell the Task DWs about any ghost cells they will need to
  // process.  This adds in entries into the task DW's d_varDB which
  // isn't a var, but is instead metadata describing how to copy ghost
  // cells between two vars listed in d_varDB.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = this->getGhostVars().getMap();
  std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it;
  for (it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    // If the neighbor is valid on the GPU, we just send in from and
    // to coordinates and call a kernel to copy those coordinates If
    // it's not valid on the GPU, we copy in the grid var and send in
    // from and to coordinates and call a kernel to copy those
    // coordinates.

    // Peer to peer GPU copies will be handled elsewhere.
    // GPU to another MPI ranks will be handled elsewhere.
    if (it->second.m_dest != GpuUtilities::anotherDeviceSameMpiRank && it->second.m_dest != GpuUtilities::anotherMpiRank) {
      int dwIndex = it->first.m_dataWarehouse;

      // We can copy it manually internally within the device via a kernel.
      // This apparently goes faster overall
      IntVector varOffset = it->second.m_varOffset;
      IntVector varSize = it->second.m_varSize;
      IntVector ghost_low = it->first.m_sharedLowCoordinates;
      IntVector ghost_high = it->first.m_sharedHighCoordinates;
      IntVector virtualOffset = it->second.m_virtualOffset;

      // Add in an entry into this Task DW's d_varDB which isn't a
      // var, but is instead metadata describing how to copy ghost
      // cells between two vars listed in d_varDB.
      this->getTaskGpuDataWarehouse(it->second.m_sourceDeviceNum, (Task::WhichDW)dwIndex)->putGhostCell(
                                                                                                        it->first.m_label.c_str(), it->second.m_sourcePatchPointer->getID(), it->second.m_destPatchPointer->getID(),
                                                                                                        it->first.m_matlIndx, it->first.m_levelIndx, it->second.m_sourceStaging, it->second.m_destStaging,
                                                                                                        make_int3(varOffset.x(), varOffset.y(), varOffset.z()), make_int3(varSize.x(), varSize.y(), varSize.z()),
                                                                                                        make_int3(ghost_low.x(), ghost_low.y(), ghost_low.z()), make_int3(ghost_high.x(), ghost_high.y(), ghost_high.z()),
                                                                                                        make_int3(virtualOffset.x(), virtualOffset.y(), virtualOffset.z()));
    }
  }
}


// ______________________________________________________________________
//
bool
DetailedTask::ghostCellsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Check that all staging data is ready for upcoming GPU to GPU
  // ghost cell copies.  Most of the time, the staging data is
  // "inside" the patch variable in the data warehouse.  But
  // sometimes, some upcoming GPU to GPU ghost cell copies will occur
  // from other patches not assigned to the task. For example, ghost
  // cell data from another MPI rank is going to be assigned to a
  // different patch, but the ghost cell data needs to be copied into
  // the patch variable assigned to this task.

  // In another example, ghost cell data may be coming from another
  // patch that is on the same MPI rank, but that other patch variable
  // is being copied host-to-GPU by another scheduler thread
  // processing another task.  The best solution here is to
  // investigate all upcoming ghost cell copies that need to occur,
  // and verify that both the source and destination patches are valid
  // in the memory space.

  // Note: I bet this is more accurate than the above check, if so,
  // remove the above loop.

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = this->getGhostVars().getMap();
  std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it;
  for (it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {

    GPUDataWarehouse* gpudw = m_dws[it->first.m_dataWarehouse]->getGPUDW(GpuUtilities::getGpuIndexForPatch(it->second.m_sourcePatchPointer));

    // Check the source
    if (it->second.m_sourceStaging) {
      if (!(gpudw->areAllStagingVarsValid(it->first.m_label.c_str(),
            it->second.m_sourcePatchPointer->getID(),
            it->second.m_matlIndx,
            it->first.m_levelIndx))) {
        return false;
      }
    } else {
      if (!(gpudw->isValidOnGPU(it->first.m_label.c_str(),
            it->second.m_sourcePatchPointer->getID(),
            it->second.m_matlIndx,
            it->first.m_levelIndx))) {
        return false;
      }
    }

    // Check the destination?
  }

  // if we got there, then everything must be ready to go.
  return true;
}


// ______________________________________________________________________
//
// TODO: Check performance. It hampered the performance by 10% for
// Poisson with patch size 4 cubed and 64 such patches. Ran with 2
// ranks, nthreads 16
bool
DetailedTask::allHostVarsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws)
{
//  printf("allHostVarsProcessingReady %s\n", this->getName().c_str());
  const Task * task = this->getTask();

  // this->clearPreparationCollections();

  std::vector<DetailedTask::labelPatchMatlLevelDw> varsNeedOnHost = this->getVarsNeededOnHost();

  for (int i = 0; i < varsNeedOnHost.size(); i++) {
    DetailedTask::labelPatchMatlLevelDw lpmld = varsNeedOnHost[i];
    OnDemandDataWarehouseP dw = m_dws[lpmld.dwIndex];
    if (!(dw->isValidOnCPU(lpmld.label.c_str(), lpmld.patchID, lpmld.matlIndx, lpmld.levelIndx ))) {
      // printf("allHostVarsProcessingReady: returns false for requires: %s %s %d %d %d \n", this->getName().c_str(),
      //       dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), patches->get(i)->getLevel()->getID());
      return false;
    }
  }

//  printf("allHostVarsProcessingReady: returns true %s\n", this->getName().c_str());
  this->getVarsNeededOnHost().clear();
  return true;
}


// ______________________________________________________________________
//
// bool
// DetailedTask::allHostVarsProcessingReady()
// {
// //  printf("allHostVarsProcessingReady %s\n", this->getName().c_str());
//  const Task * task = this->getTask();
//
//  this->clearPreparationCollections();
//
//  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
//    const TypeDescription::Type type = dependantVar->m_var->typeDescription()->getType();
//    if (type == TypeDescription::CCVariable ||
//        type == TypeDescription::NCVariable ||
//        type == TypeDescription::SFCXVariable ||
//        type == TypeDescription::SFCYVariable ||
//        type == TypeDescription::SFCZVariable)
//    {
//      int dwIndex = dependantVar->mapDataWarehouse();
//      OnDemandDataWarehouseP dw = m_dws[dwIndex];
//      constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
//      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
//      const int numPatches = patches->size();
//      const int numMatls = matls->size();
//      for (int i = 0; i < numPatches; i++) {
//        for (int j = 0; j < numMatls; j++) {
//          if (!(dw->isValidOnCPU(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), patches->get(i)->getLevel()->getID()))) {
// //            printf("allHostVarsProcessingReady: returns false for requires: %s %s %d %d %d \n", this->getName().c_str(),
// //                            dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), patches->get(i)->getLevel()->getID());
//            return false;
//          }
//        }
//      }
//    }
//  }
//
//  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
//    const TypeDescription::Type type = dependantVar->m_var->typeDescription()->getType();
//    if (type == TypeDescription::CCVariable ||
//        type == TypeDescription::NCVariable ||
//        type == TypeDescription::SFCXVariable ||
//        type == TypeDescription::SFCYVariable ||
//        type == TypeDescription::SFCZVariable)
//    {
//      int dwIndex = dependantVar->mapDataWarehouse();
//      OnDemandDataWarehouseP dw = m_dws[dwIndex];
//      constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
//      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
//      const int numPatches = patches->size();
//      const int numMatls = matls->size();
//      for (int i = 0; i < numPatches; i++) {
//        for (int j = 0; j < numMatls; j++) {
//          if (!(dw->isValidOnCPU(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), patches->get(i)->getLevel()->getID()))) {
// //            printf("allHostVarsProcessingReady: returns false for modifies: %s %s %d %d %d \n", this->getName().c_str(),
// //                    dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), patches->get(i)->getLevel()->getID());
//            return false;
//          }
//        }
//      }
//    }
//  }
// //  printf("allHostVarsProcessingReady: returns true %s\n", this->getName().c_str());
//  return true;
// }


// bool
// DetailedTask::allHostVarsProcessingReady()
// {
//
//  const Task* task = this->getTask();
//
//  this->clearPreparationCollections();
//
//  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
//  // transfer some variables twice).
//  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
//  // materials, but a single task will never run multiple levels.
//  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
//  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
//    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
//    if (patches) {
//      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
//      const int numPatches = patches->size();
//      const int numMatls = matls->size();
//      for (int i = 0; i < numPatches; i++) {
//        for (int j = 0; j < numMatls; j++) {
//          labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
//          if (vars.find(lpmd) == vars.end()) {
//            vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
//          }
//        }
//      }
//    } else {
//      std::cout << myRankThread() << " In allHostVarsProcessingReady, no patches, task is " << this->getName() << std::endl;
//    }
//  }
//
//  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
//    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
//    if (patches) {
//      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
//      const int numPatches = patches->size();
//      const int numMatls = matls->size();
//      for (int i = 0; i < numPatches; i++) {
//        for (int j = 0; j < numMatls; j++) {
//          labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Modifies);
//          if (vars.find(lpmd) == vars.end()) {
//            vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
//          }
//        }
//      }
//    } else {
//      std::cout << myRankThread() << " In allHostVarsProcessingReady, no patches, task is " << this->getName() << std::endl;
//    }
//  }
//
//  // Go through each var, see if it's valid or valid with ghosts.
//  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
//  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
//    const Task::Dependency* curDependency = varIter->second;
//
//    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(this->getPatches());
//    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(this->getMaterials());
//    const int numPatches = patches->size();
//    const int patchID = varIter->first.m_patchID;
//    const Patch * patch = nullptr;
//    const Level * level = nullptr;
//    for (int i = 0; i < numPatches; i++) {
//      if (patches->get(i)->getID() == patchID) {
//        patch = patches->get(i);
//        level = patch->getLevel();
//      }
//    }
//    int levelID = level->getID();
//    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
//      levelID = -1;
//    }
//    const int matlID = varIter->first.m_matlIndex;
//    const int dwIndex = curDependency->mapDataWarehouse();
//    OnDemandDataWarehouseP dw = m_dws[dwIndex];
//    GPUDataWarehouse* gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
//    if (curDependency->m_dep_type == Task::Requires || curDependency->m_dep_type == Task::Modifies) {
//      if (gpudw->dwEntryExistsOnCPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID)) {
//        if (!(gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID))) {
//          return false;
//        }
//      }
//    }
//  }
//
//  return true;
// }

// ______________________________________________________________________
//
bool
DetailedTask::allGPUVarsProcessingReady(std::vector<OnDemandDataWarehouseP> & m_dws)
{

  const Task* task = this->getTask();

  // this->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and
  // remove duplicates (we don't want to transfer some variables
  // twice).

  // Note: A task can only run on one level at a time.  It could run
  // multiple patches and multiple materials, but a single task will
  // never run multiple levels.
  std::multimap<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        std::multimap<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter = vars.find(lpmd);
        if (varIter == vars.end() || varIter->second->mapDataWarehouse() != dependantVar->mapDataWarehouse()) {
          vars.insert(std::multimap<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Modifies);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  std::multimap<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* curDependency = varIter->second;

    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::SoleVariable) {
      continue; // TODO: We are ignoring SoleVariables for now. They
                // should be managed.
    }

    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch * patch = nullptr;
    const Level * level = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
        level = patch->getLevel();
      }
    }
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }

    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse* gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires || curDependency->m_dep_type == Task::Modifies) {
      if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {
        // it has ghost cells.
        if (!(gpudw->isValidWithGhostsOnGPU(curDependency->m_var->getName().c_str(),patchID, matlID, levelID)) ||
            !(gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(),patchID, matlID, levelID))
        ) {
          return false;
        }
      } else {
        // If it's a gridvar, then we just don't have the ghost cells
        // processed yet by another thread If it's another type of
        // variable, something went wrong, it should have been marked
        // as valid previously.
        if (!(gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(),patchID, matlID, levelID))) {
          return false;
        }
      }
    }
  }

  // if we got there, then everything must be ready to go.
  return true;
}


// ______________________________________________________________________
//
void
DetailedTask::markDeviceRequiresAndModifiesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // This marks any Requires and Modifies variable as valid that
  // wasn't in the GPU but is now in the GPU.

  // If they were already in the GPU due to being computes from a
  // previous time step, it was already marked as valid.  So there is
  // no need to do anything extra for them.

  // If they weren't in the GPU yet, this task or another task copied
  // it in.

  // If it's another task that copied it in, we let that task manage
  // it.

  // If it was this task, then those variables which this task copied
  // in are found in varsBeingCopiedByTask.

  // By the conclusion of this method, some variables will be valid
  // and awaiting ghost cells, some will just be valid if they had no
  // ghost cells, and some variables will be undetermined if they're
  // being managed by another task.

  // After this method, a kernel is invoked to process ghost cells.


  // Go through device requires vars and mark them as valid on the
  // device.  They are either already valid because they were there
  // previously.  Or they just got copied in and the stream completed.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = this->getVarsBeingCopiedByTask().getMap();
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
            it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires || it->second.m_dep->m_dep_type == Task::Modifies) {
      bool success=false;
      if (!it->second.m_staging) {
        success = gpudw->compareAndSwapSetValidOnGPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
      } else {
        success = gpudw->compareAndSwapSetValidOnGPUStaging(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx,
                                    make_int3(it->second.m_offset.x(),it->second.m_offset.y(),it->second.m_offset.z()),
                                    make_int3(it->second.m_sizeVector.x(), it->second.m_sizeVector.y(), it->second.m_sizeVector.z()));
      }

      if(success){      // release only if SetValud returns
                        // true. Otherwise double deletion (probably
                        // due to race condition) and then segfault
                        // was observed in dqmom example
        if (it->second.m_tempVarToReclaim) {
          // Release our reference to the variable data that
          // getGridVar returned
          delete it->second.m_tempVarToReclaim;
        }
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::markDeviceGhostsAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Go through requires vars and mark them as valid on the device.
  // They are either already valid because they were there previously.
  // Or they just got copied in and the stream completed.  Now go
  // through the varsToBeGhostReady collection.  Any in there should
  // be marked as valid with ghost cells
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = this->getVarsToBeGhostReady().getMap();
  for (auto it = varMap.begin(); it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);

    gpudw->setValidWithGhostsOnGPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
  }
}


// ______________________________________________________________________
//
void
DetailedTask::markHostComputesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Go through device computes vars and mark them as valid on the device.
  // std::lock_guard<Uintah::MasterLock> race(race_lock);
  // The only thing we need to process is the requires.
  const Task* task = this->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(this->getMaterials());
    // this is so we can allocate persistent events and streams to
    // distribute when needed one stream and one event per variable
    // per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      for (int j = 0; j < numMatls; j++) {
        int patchID = patches->get(i)->getID();
        int matlID = matls->get(j);
        const Level* level = patches->get(i)->getLevel();
        int levelID = level->getID();
        if (gpudw && gpudw->isAllocatedOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID)) {
          gpudw->compareAndSwapSetInvalidOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          gpudw->compareAndSwapSetInvalidWithGhostsOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          // DS: not using gpudw->compareAndSwapSetValidOnCPU here
          // because entry will not be there in GPU dw.
        }
        dw->compareAndSwapSetValidOnCPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
        dw->compareAndSwapSetInvalidWithGhostsOnCPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::markDeviceComputesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Go through device computes vars and mark them as valid on the
  // device.  The only thing we need to process is the requires.
  const Task* task = this->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(this->getMaterials());
    // this is so we can allocate persistent events and streams to
    // distribute when needed one stream and one event per variable
    // per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      for (int j = 0; j < numMatls; j++) {
        int patchID = patches->get(i)->getID();
        int matlID = matls->get(j);
        const Level* level = patches->get(i)->getLevel();
        int levelID = level->getID();
        if (gpudw && gpudw->isAllocatedOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID)) {
          gpudw->compareAndSwapSetValidOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          gpudw->compareAndSwapSetInvalidWithGhostsOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          gpudw->compareAndSwapSetInvalidOnCPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
        }
        dw->compareAndSwapSetInvalidOnCPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
        dw->compareAndSwapSetInvalidWithGhostsOnCPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::markDeviceModifiesGhostAsInvalid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Go through device modifies vars and mark ghosts as invalid.
  const Task* task = this->getTask();
  for (const Task::Dependency* comp = task->getModifies(); comp != 0; comp = comp->m_next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(this->getMaterials());
    // this is so we can allocate persistent events and streams to
    // distribute when needed one stream and one event per variable
    // per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      if (gpudw != nullptr) {
        for (int j = 0; j < numMatls; j++) {
          int patchID = patches->get(i)->getID();
          int matlID = matls->get(j);
          const Level* level = patches->get(i)->getLevel();
          int levelID = level->getID();
          if (gpudw->isAllocatedOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID)) {
            gpudw->compareAndSwapSetInvalidWithGhostsOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          }
        }
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::markDeviceAsInvalidHostAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Data has been copied from the device to the host.  The stream has
  // completed.  Go through all variables that this CPU task was
  // responsible for copying mark them as valid on the CPU

  // DS: 10252019 if upcoming task modifies data on host, set the
  // variable invalid in gpu dw.  This ensures a H2D copy if any
  // subsequent task requires/modifies variable on device.

  // TODO: check is it needed for all type of variables?

  // CAUTION: Positioning of compareAndSwapSetInvalidOnCPU/GPU methods
  // is very sensitive.  Wrong placement can make the variable invalid
  // on both execution spaces and then task runner loop just hangs. Be
  // extremely careful of placing code.  The only thing we need to
  // process is the modifies.  var maps associated with detailed task
  // do not ALWAYs contain variable - especially if task is a host
  // task.  So use raw dependency linked lists from task rather than
  // var maps associated with detailed task.
  const Task * task = this->getTask();
  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        const Patch * patch = patches->get(i);
        unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
        int dwIndex = dependantVar->mapDataWarehouse();
        OnDemandDataWarehouseP dw = m_dws[dwIndex];
        GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);

        const char * var_name = dependantVar->m_var->getName().c_str();
        int patchID = patch->getID();
        int matlID = matls->get(j);
        int levelID = patch->getLevel()->getID();

        // modified on CPU. mark device as invalid and host as valid
        gpudw->compareAndSwapSetInvalidOnGPU(var_name, patchID, matlID, levelID);
        gpudw->compareAndSwapSetInvalidWithGhostsOnGPU(var_name, patchID, matlID, levelID);

        // modified on CPU. mark host as valid, but host ghost as invalid
        if(gpudw->dwEntryExists(var_name, patchID, matlID, levelID)){
          gpudw->compareAndSwapSetValidOnCPU(var_name, patchID, matlID, levelID);
        }
        dw->compareAndSwapSetValidOnCPU(var_name, patchID, matlID, levelID);
        dw->compareAndSwapSetInvalidWithGhostsOnCPU(var_name, patchID, matlID, levelID);
      }
    }
  }
}


// ______________________________________________________________________
//
// dont mark device as valid here because computation on device might not be yet completed.
void
DetailedTask::markHostAsInvalid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Data has been copied from the device to the host.  The stream has
  // completed.  Go through all variables that this CPU task was
  // responsible for copying mark them as valid on the CPU

  // DS: 10252019 if upcoming task modifies data on host, set the
  // variable invalid in gpu dw.  This ensures a H2D copy if any
  // subsequent task requires/modifies variable on device.

  // TODO: check is it needed for all type of variables?

  // CAUTION: Positioning of compareAndSwapSetInvalidOnCPU/GPU methods
  // is very sensitive.  Wrong placement can make the variable invalid
  // on both execution spaces and then task runner loop just hangs. Be
  // extremely careful of placing code.  The only thing we need to
  // process is the modifies.
  // std::multimap<GpuUtilities::LabelPatchMatlLevelDw,
  // DeviceGridVariableInfo> & varMap =
  // this->getVarsBeingCopiedByTask().getMap();

  // var maps associated with detailed task do not ALWAYs contain
  // variable - especially if task is a host task.  So use raw
  // dependency linked lists from task rather than var maps associated
  // with detailed task.
  const Task * task = this->getTask();
  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        const Patch * patch = patches->get(i);
        unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
        int dwIndex = dependantVar->mapDataWarehouse();
        OnDemandDataWarehouseP dw = m_dws[dwIndex];
        GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);

        const char * var_name = dependantVar->m_var->getName().c_str();
        int patchID = patch->getID();
        int matlID = matls->get(j);
        int levelID = patch->getLevel()->getID();

        gpudw->compareAndSwapSetInvalidOnCPU(var_name, patchID, matlID, levelID);
        dw->compareAndSwapSetInvalidOnCPU(var_name, patchID, matlID, levelID);
        dw->compareAndSwapSetInvalidWithGhostsOnCPU(var_name, patchID, matlID, levelID);
      }
    }
  }
}


// ______________________________________________________________________
//
// hoping that this will ALWAYS follow call to markHostAsInvalid after GPU modifies / computes. Otherwise the application will hang
void
DetailedTask::markHostRequiresAndModifiesDataAsValid(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Data has been copied from the device to the host.  The stream has
  // completed.  Go through all variables that this CPU task was
  // responsible for copying mark them as valid on the CPU

  // The only thing we need to process is the requires.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = this->getVarsBeingCopiedByTask().getMap();
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
            it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires || it->second.m_dep->m_dep_type == Task::Modifies) {
      if (!it->second.m_staging) {
        if(gpudw->dwEntryExists(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx)){
          gpudw->compareAndSwapSetValidOnCPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
        }
        m_dws[dwIndex]->compareAndSwapSetValidOnCPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
        m_dws[dwIndex]->compareAndSwapSetInvalidWithGhostsOnCPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
      }
      if (it->second.m_var) {
        // Release our reference to the variable data that getGridVar returned
        delete it->second.m_var;
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::initiateD2HForHugeGhostCells(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // RMCRT problems use 32768 ghost cells as a way to force an "all to
  // all" transmission of ghost cells It is much easier to manage
  // these ghost cells in host memory instead of GPU memory.  So for
  // such variables, after they are done computing, we will copy them
  // D2H.  For RMCRT, this overhead only adds about 1% or less to the
  // overall computation time.  '

  // This only works with COMPUTES, it is not configured to work with
  // requires.

  const Task* task = this->getTask();

  // determine which computes variables to copy back to the host
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
    // Only process large number of ghost cells.
    if (comp->m_num_ghost_cells == SHRT_MAX) {
      constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(this->getPatches());
      constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(this->getMaterials());

      int dwIndex = comp->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];

      void* host_ptr   = nullptr;    // host base pointer to raw data
      void* device_ptr = nullptr;    // device base pointer to raw data
      size_t host_bytes = 0;         // raw byte count to copy to the device

      IntVector host_low, host_high, host_offset, host_size, host_strides;

      int numPatches = patches->size();
      int numMatls = matls->size();
      // __________________________________
      //
      for (int i = 0; i < numPatches; ++i) {
        for (int j = 0; j < numMatls; ++j) {
          const int patchID = patches->get(i)->getID();
          const int matlID  = matls->get(j);

          const std::string compVarName = comp->m_var->getName();

          const Patch * patch = nullptr;
          const Level* level = nullptr;
          for (int i = 0; i < numPatches; i++) {
            if (patches->get(i)->getID() == patchID) {
             patch = patches->get(i);
             level = patch->getLevel();
            }
          }
          if (!patch) {
           printf("ERROR:\nDetailedTask::initiateD2HForHugeGhostCells() patch not found.\n");
           SCI_THROW( InternalError("DetailedTask::initiateD2HForHugeGhostCells() patch not found.", __FILE__, __LINE__));
          }
          const int levelID = level->getID();

          const unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
          GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);

          // Base call is commented out
          // OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum);

          if (gpudw != nullptr) {

            // It's not valid on the CPU but it is on the GPU.  Copy it on over.
            if (!gpudw->isValidOnCPU( compVarName.c_str(), patchID, matlID, levelID)) {
              const TypeDescription::Type type = comp->m_var->typeDescription()->getType();
              const TypeDescription::Type datatype = comp->m_var->typeDescription()->getSubType()->getType();
              switch (type) {
                case TypeDescription::CCVariable:
                case TypeDescription::NCVariable:
                case TypeDescription::SFCXVariable:
                case TypeDescription::SFCYVariable:
                case TypeDescription::SFCZVariable: {

                  bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(compVarName.c_str(), patchID, matlID, levelID);
                  if (performCopy) {
                    // size the host var to be able to fit all room needed.
                    IntVector host_low, host_high, host_lowOffset, host_highOffset, host_offset, host_size, host_strides;
                    level->computeVariableExtents(type, host_low, host_high);
                    int dwIndex = comp->mapDataWarehouse();
                    OnDemandDataWarehouseP dw = m_dws[dwIndex];

                    // It's possible the computes data may contain
                    // ghost cells.  But a task needing to get the
                    // data out of the GPU may not know this.  It may
                    // just want the var data.  This creates a
                    // dilemma, as the GPU var is sized differently
                    // than the CPU var.  So ask the GPU what size it
                    // has for the var.  Size the CPU var to match so
                    // it can copy all GPU data in.  When the GPU->CPU
                    // copy is done, then we need to resize the CPU
                    // var if needed to match what the CPU is
                    // expecting it to be.

                    // GPUGridVariableBase* gpuGridVar;

                    int3 low;
                    int3 high;
                    int3 size;
                    GPUDataWarehouse::GhostType tempgtype;
                    Ghost::GhostType gtype;
                    int numGhostCells;
                    gpudw->getSizes(low, high, size, tempgtype, numGhostCells, compVarName.c_str(), patchID, matlID, levelID);

                    gtype = (Ghost::GhostType) tempgtype;

                    GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(comp->m_var->typeDescription()->createInstance());

                    bool finalized = dw->isFinalized();
                    if (finalized) {
                      dw->unfinalize();
                    }

                    dw->allocateAndPut(*gridVar, comp->m_var, matlID, patch, gtype, numGhostCells);
                    if (finalized) {
                      dw->refinalize();
                    }

                    gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
                    host_ptr = gridVar->getBasePointer();
                    host_bytes = gridVar->getDataSize();

                    int3 device_offset;
                    int3 device_size;
                    GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(datatype);
                    gpudw->get(*device_var, compVarName.c_str(), patchID, matlID, levelID);
                    device_var->getArray3(device_offset, device_size, device_ptr);
                    delete device_var;

                    // if offset and size is equal to CPU DW, directly
                    // copy back to CPU var memory;
                    if (   device_offset.x == host_low.x()
                        && device_offset.y == host_low.y()
                        && device_offset.z == host_low.z()
                        && device_size.x   == host_size.x()
                        && device_size.y   == host_size.y()
                        && device_size.z   == host_size.z()) {

#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
                      m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
                      m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                                deviceNum,
                                                host_ptr, device_ptr,
                                                host_bytes, cudaMemcpyDeviceToHost);
#else
                      cudaStream_t* stream = this->getCudaStreamForThisTask(deviceNum);
                      cudaError_t retVal;
                      // ARS cudaMemcpyAsync copy from device to host
                      // Kokkos equivalent - deep copy.
                      CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
#endif

                      IntVector temp(0,0,0);
                      this->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                            false,
                                                            IntVector(device_size.x, device_size.y, device_size.z),
                                                            host_strides.x(), host_bytes,
                                                            IntVector(device_offset.x, device_offset.y, device_offset.z),
                                                            comp,
                                                            gtype, numGhostCells,  deviceNum,
                                                            gridVar, GpuUtilities::sameDeviceSameMpiRank);


                      // ARS cuda error handling. If there is an error
                      // not sure the code would get here as the
                      // previous call to CUDA_RT_SAFE_CALL would exit
                      // upon an error.
                      // Kokkos equivalent - not needed??
                      // if (retVal == cudaErrorLaunchFailure) {
                      //   SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ this->getName(), __FILE__, __LINE__));
                      // } else {
                      //   CUDA_RT_SAFE_CALL(retVal);
                      // }
                    }
                    delete gridVar;
                  }
                  break;
                }
                default:
                  std::ostringstream warn;
                  warn << "  ERROR: DetailedTask::initiateD2HForHugeGhostCells (" << this->getName() << ") variable: "
                       << comp->m_var->getName() << " not implemented " << std::endl;
                  SCI_THROW(InternalError( warn.str() , __FILE__, __LINE__));

              }
            }
          }
        }
      }
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::initiateD2H( const ProcessorGroup                * d_myworld,
                           std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Request that all contiguous device arrays from the device be sent
  // to their contiguous host array counterparts.  We only copy back
  // the data needed for an upcoming task.  If data isn't needed, it
  // can stay on the device and potentially even die on the device

  // Returns true if no device data is required, thus allowing a CPU
  // task to immediately proceed.

  void* host_ptr    = nullptr;   // host base pointer to raw data
  void* device_ptr  = nullptr;   // device base pointer to raw data
  size_t host_bytes = 0;         // raw byte count to copy to the device

  const Task* task = this->getTask();
  this->clearPreparationCollections();
  this->getVarsNeededOnHost().clear();

  std::vector<const Patch *> validNeighbourPatches;

  auto m_loadBalancer = m_task_group->getSchedulerCommon()->getLoadBalancer();

  // The only thing we need to process is the requires.  Gather up all
  // possible dependents and remove duplicate (we don't want to
  // transfer some variables twice)
  std::multimap<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator iter = vars.find(lpmd);
        if (iter == vars.end() || iter->second->m_whichdw != dependantVar->m_whichdw) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));

          const TypeDescription::Type type = dependantVar->m_var->typeDescription()->getType();

          if ( type == TypeDescription::CCVariable   ||
               type == TypeDescription::NCVariable   ||
               type == TypeDescription::SFCXVariable ||
               type == TypeDescription::SFCYVariable ||
               type == TypeDescription::SFCZVariable
             )
          {
            int dwIndex = dependantVar->mapDataWarehouse();
            OnDemandDataWarehouseP dw = m_dws[dwIndex];
            std::vector<OnDemandDataWarehouse::ValidNeighbors> validNeighbors;
            dw->getValidNeighbors(dependantVar->m_var, matls->get(j), patches->get(i), dependantVar->m_gtype, dependantVar->m_num_ghost_cells, validNeighbors, true);

            for (std::vector<OnDemandDataWarehouse::ValidNeighbors>::iterator iter = validNeighbors.begin(); iter != validNeighbors.end(); ++iter) {
              const Patch * sourcePatch;

              if (iter->neighborPatch->getID() >= 0) {
                sourcePatch = iter->neighborPatch;
              } else {
                // This occurs on virtual patches.  They can be "wrap
                // around" patches, meaning if you go to one end of a
                // domain you will show up on the other side.  Virtual
                // patches have negative patch IDs, but they know what
                // real patch they are referring to.
                sourcePatch = iter->neighborPatch->getRealPatch();
              }

              // TODO: handle virtual patch case.
              if (m_loadBalancer->getPatchwiseProcessorAssignment(sourcePatch) == d_myworld->myRank()) { // only if its a local patch
                validNeighbourPatches.push_back(sourcePatch);
                labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), sourcePatch->getID(), matls->get(j), Task::Requires);
                if (vars.find(lpmd) == vars.end()) {
                  vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
                }
              }
            }
          }
        }
      }
    }
  }

  for (const Task::Dependency* dependantVar = task->getModifies(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Modifies);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  for (const Task::Dependency* dependantVar = task->getComputes(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Computes);
      }
    }
  }

  // Go through each unique dependent var and see if we should queue
  // up a D2H copy.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* dependantVar = varIter->second;
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(this->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(this->getMaterials());


    // This is so we can allocate persistent events and streams to
    // distribute when needed one stream and one event per variable
    // per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int dwIndex = dependantVar->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    // Find the patch and level objects associated with the patchID
    const int patchID = varIter->first.m_patchID;
    bool isNeighbor = false;
    const Patch * patch = nullptr;
    const Level * level = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
        level = patch->getLevel();
      }
    }

    if (!patch) {
      for (size_t i = 0; i < validNeighbourPatches.size(); i++) {
        if (validNeighbourPatches[i]->getID() == patchID) {
          patch = validNeighbourPatches[i];
          level = patch->getLevel();
          isNeighbor = true;
        }
      }
    }

    if (!patch) {
      printf("ERROR:\nDetailedTask::initiateD2H() patch not found.\n");
      SCI_THROW( InternalError("DetailedTask::initiateD2H() patch not found.", __FILE__, __LINE__));
    }

    int levelID = level->getID();
    if (dependantVar->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }

    const int matlID = varIter->first.m_matlIndex;

    int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);

    if (deviceNum < 0) { // patch is not present on any of the devices.
      if (isNeighbor) {  // Sometimes getValidNeighbors returns
                         // patches out of the domain (do not know
                         // why). In such cases do not raise
                         // error. Its a hack.
        continue;
      } else {
        cerrLock.lock();
        {
           std::cerr << "ERROR: Could not find the assigned GPU for this patch. patch: " << patch->getID() <<
           " " << __FILE__ << ":" << __LINE__ << std::endl;
        }
        cerrLock.unlock();
        exit(-1);
      }
    }

    GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);

    // Base call is commented out
    // OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum);

    const std::string varName = dependantVar->m_var->getName();

    // TODO: Titan production hack.  A clean hack, but should be fixed. Brad P Dec 1 2016

    // There currently exists a race condition.  Suppose cellType is in both host and GPU
    // memory.  Currently the GPU data warehouse knows it is in GPU memory, but it doesn't
    // know if it's in host memory (the GPU DW doesn't track lifetimes of host DW vars).
    // Thread 2 - Task A requests a requires var for cellType for the host newDW, and gets it.
    // Thread 3 - Task B invokes the initiateD2H check, thinks there is no host instance of cellType,
    //            so it initiates a D2H, which performs another host allocateAndPut, and the subsequent put
    //            deletes the old entry and creates a new entry.
    // Race condition is that thread 2's pointer has been cleaned up, while thread 3 has a new one.
    // A temp fix could be to check if all host vars exist in the host dw prior to launching the task.
    // For now, the if statement's job is to ignore a GPU task's *requires* that nay get pulled D2H
    // by subsequent CPU tasks.  For example, RMCRT computes divQ, RMCRTboundFlux, and radiationVolq.
    // and requires other variables.  So the logic is "If it wasn't one of the computes", then we
    // don't need to copy it back D2H"

    // DS 04222020: Commented hack_foundAComputes hack as now CPU
    // validity status of a variable is maintained in OnDemandWH and
    // whether D2H copy is needed can be determined dynamically.
//    bool hack_foundAComputes{false};
//
//    // RMCRT hack:
//    if ( (varName == "divQ")           ||
//         (varName == "RMCRTboundFlux") ||
//         (varName == "radiationVolq")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // almgren-mmsBC.ups hack
//    // almgren-mms_conv.ups hack
//    if ( (varName == "uVelocity") ||
//         (varName == "vVelocity") ||
//         (varName == "wVelocity")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // box1.ups hack
//    if ( (varName == "length_0_x_dflux") ||
//         (varName == "length_0_y_dflux") ||
//         (varName == "length_0_z_dflux") ||
//         (varName == "length_1_x_dflux") ||
//         (varName == "length_1_y_dflux") ||
//         (varName == "length_1_z_dflux") ||
//         (varName == "pU_0_x_dflux")     ||
//         (varName == "pU_0_y_dflux")     ||
//         (varName == "pU_0_z_dflux")     ||
//         (varName == "pV_0_x_dflux")     ||
//         (varName == "pV_0_y_dflux")     ||
//         (varName == "pV_0_z_dflux")     ||
//         (varName == "pW_0_x_dflux")     ||
//         (varName == "pW_0_y_dflux")     ||
//         (varName == "pW_0_z_dflux")     ||
//         (varName == "pU_1_x_dflux")     ||
//         (varName == "pU_1_y_dflux")     ||
//         (varName == "pU_1_z_dflux")     ||
//         (varName == "pV_1_x_dflux")     ||
//         (varName == "pV_1_y_dflux")     ||
//         (varName == "pV_1_z_dflux")     ||
//         (varName == "pW_1_x_dflux")     ||
//         (varName == "pW_1_y_dflux")     ||
//         (varName == "pW_1_z_dflux")     ||
//         (varName == "w_qn0_x_dflux")    ||
//         (varName == "w_qn0_y_dflux")    ||
//         (varName == "w_qn0_z_dflux")    ||
//         (varName == "w_qn1_x_dflux")    ||
//         (varName == "w_qn1_y_dflux")    ||
//         (varName == "w_qn1_z_dflux")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // dqmom_example_char_no_pressure.ups hack:
//    // All the computes are char_ps_qn4, char_ps_qn4_gasSource, char_ps_qn4_particletempSource, char_ps_qn4_particleSizeSource
//    // char_ps_qn4_surfacerate, char_gas_reaction0_qn4, char_gas_reaction1_qn4, char_gas_reaction2_qn4.  Note that the qn# goes
//    // from qn0 to qn4.  Also, the char_gas_reaction0_qn4 variable is both a computes in the newDW and a requires in the oldDW
//    if ( (varName.substr(0,10) == "char_ps_qn")                                  ||
//         (varName.substr(0,17) == "char_gas_reaction" && dwIndex == Task::NewDW) ||
//         (varName == "raw_coal_0_x_dflux")                                       ||
//         (varName == "raw_coal_0_y_dflux")                                       ||
//         (varName == "raw_coal_1_x_dflux")                                       ||
//         (varName == "raw_coal_1_y_dflux")                                       ||
//         (varName == "raw_coal_1_z_dflux")                                       ||
//         (varName == "w_qn2_x_dflux")                                            ||
//         (varName == "w_qn2_y_dflux")                                            ||
//         (varName == "w_qn3_x_dflux")                                            ||
//         (varName == "w_qn4_x_dflux")                                            ||
//         (varName == "w_qn4_y_dflux")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // heliumKS_pressureBC.ups hack
//    if ( (varName == "A_press")            ||
//         (varName == "b_press")            ||
//         (varName == "cellType")           ||
//         (varName == "continuity_balance") ||
//         (varName == "density")            ||
//         (varName == "density_star")       ||
//         (varName == "drhodt")             ||
//         (varName == "gamma")              ||
//         (varName == "gravity_z")          ||
//         (varName == "gridX")              ||
//         (varName == "gridY")              ||
//         (varName == "gridZ")              ||
//         (varName == "guess_press")        ||
//         (varName == "phi")                ||
//         (varName == "phi_x_dflux")        ||
//         (varName == "phi_y_dflux")        ||
//         (varName == "phi_z_dflux")        ||
//         (varName == "phi_x_flux")         ||
//         (varName == "phi_y_flux")         ||
//         (varName == "phi_z_flux")         ||
//         (varName == "pressure")           ||
//         (varName == "rho_phi")            ||
//         (varName == "rho_phi_RHS")        ||
//         (varName == "sigma11")            ||
//         (varName == "sigma12")            ||
//         (varName == "sigma13")            ||
//         (varName == "sigma22")            ||
//         (varName == "sigma23")            ||
//         (varName == "sigma33")            ||
//         (varName == "t_viscosity")        ||
//         (varName == "ucell_xvel")         ||
//         (varName == "ucell_yvel")         ||
//         (varName == "ucell_zvel")         ||
//         (varName == "vcell_xvel")         ||
//         (varName == "vcell_yvel")         ||
//         (varName == "vcell_zvel")         ||
//         (varName == "wcell_xvel")         ||
//         (varName == "wcell_yvel")         ||
//         (varName == "wcell_zvel")         ||
//         (varName == "ucellX")             ||
//         (varName == "vcellY")             ||
//         (varName == "wcellZ")             ||
//         (varName == "uVel")               ||
//         (varName == "vVel")               ||
//         (varName == "wVel")               ||
//         (varName == "uVel_cc")            ||
//         (varName == "vVel_cc")            ||
//         (varName == "wVel_cc")            ||
//         (varName == "volFraction")        ||
//         (varName == "volFractionX")       ||
//         (varName == "volFractionY")       ||
//         (varName == "volFractionZ")       ||
//         (varName == "x-mom")              ||
//         (varName == "x-mom_RHS")          ||
//         (varName == "x-mom_x_flux")       ||
//         (varName == "x-mom_y_flux")       ||
//         (varName == "x-mom_z_flux")       ||
//         (varName == "y-mom")              ||
//         (varName == "y-mom_RHS")          ||
//         (varName == "z-mom")              ||
//         (varName == "z-mom_RHS")          ||
//         (varName == "z-mom_x_flux")       ||
//         (varName == "z-mom_y_flux")       ||
//         (varName == "z-mom_z_flux")       ||
//         (varName == "hypre_solver_label")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // isotropic_kokkos_dynSmag_unpacked_noPress.ups hack:
//    if ( (varName == "uVelocity_cc") ||
//         (varName == "vVelocity_cc") ||
//         (varName == "wVelocity_cc")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // isotropic_kokkos_wale.ups hack:
//    if ( (varName == "wale_model_visc")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    // poisson1.ups hack:
//    if ( (varName == "phi")      ||
//         (varName == "residual")
//       )
//    {
//      hack_foundAComputes = true;
//    }
//
//    if (g_d2h_dbg) {
//      std::ostringstream message;
//      message << "  " << varName << ": Device-to-Host Copy May Be Needed";
//      DOUT(true, message.str());
//    }
//
//    if (!hack_foundAComputes) {
//      continue; // This variable wasn't a computes, we shouldn't do a device-to-host transfer
//                // Go start the loop over and get the next potential variable.
//    }

    if (gpudw != nullptr) {
      // It's not valid on the CPU but it is on the GPU.  Copy it on over.
      if (!dw->isValidOnCPU( varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isAllocatedOnGPU( varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isValidOnGPU( varName.c_str(), patchID, matlID, levelID)) {

        DetailedTask::labelPatchMatlLevelDw lpmld(varName.c_str(), patchID, matlID, levelID, dwIndex);
        this->getVarsNeededOnHost().push_back(lpmld);

        const TypeDescription::Type type = dependantVar->m_var->typeDescription()->getType();
        const TypeDescription::Type datatype = dependantVar->m_var->typeDescription()->getSubType()->getType();
        switch (type) {
          case TypeDescription::CCVariable:
          case TypeDescription::NCVariable:
          case TypeDescription::SFCXVariable:
          case TypeDescription::SFCYVariable:
          case TypeDescription::SFCZVariable: {

            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {

              // It's possible the computes data may contain ghost
              // cells.  But a task needing to get the data out of the
              // GPU may not know this.  It may just want the var
              // data.  This creates a dilemma, as the GPU var is
              // sized differently than the CPU var.  So ask the GPU
              // what size it has for the var.  Size the CPU var to
              // match so it can copy all GPU data in.  When the
              // GPU->CPU copy is done, then we need to resize the CPU
              // var if needed to match what the CPU is expecting it
              // to be.

              // Get the host var variable
              GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const size_t elementDataSize =
                  OnDemandDataWarehouse::getTypeDescriptionSize(dependantVar->m_var->typeDescription()->getSubType()->getType());

              // The device will have our best knowledge of the exact
              // dimensions/ghost cells of the variable, so lets get
              // those values.
              int3 device_low;
              int3 device_offset;
              int3 device_high;
              int3 device_size;
              GPUDataWarehouse::GhostType tempgtype;
              Ghost::GhostType gtype;
              int numGhostCells;
              gpudw->getSizes(device_low, device_high, device_size, tempgtype, numGhostCells, varName.c_str(), patchID, matlID, levelID);
              gtype = (Ghost::GhostType) tempgtype;
              device_offset = device_low;

              // Now get dimensions for the host variable.
              bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);
              Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);

              // Get the size/offset of what the host var would be
              // with ghost cells.
              IntVector host_low, host_high, host_lowOffset, host_highOffset, host_offset, host_size;
              if (uses_SHRT_MAX) {
                level->findCellIndexRange(host_low, host_high); // including extraCells
              } else {
                // DS 12132019: GPU Resize fix. Use max ghost cells
                // and corresponding low and high to allocate scratch
                // space
                Patch::getGhostOffsets(type, dependantVar->m_var->getMaxDeviceGhostType(), dependantVar->m_var->getMaxDeviceGhost(), host_lowOffset, host_highOffset);
                patch->computeExtents(basis, dependantVar->m_var->getBoundaryLayer(), host_lowOffset, host_highOffset, host_low, host_high);
              }
              host_size = host_high - host_low;
              int dwIndex = dependantVar->mapDataWarehouse();
              OnDemandDataWarehouseP dw = m_dws[dwIndex];

              // Get the device var so we can get the pointer.
              GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(datatype);
              gpudw->get(*device_var, varName.c_str(), patchID, matlID, levelID);
              device_var->getArray3(device_offset, device_size, device_ptr);
              delete device_var;

              bool proceedWithCopy = false;
              // See if the size of the host var and the device var match.

              if (   device_offset.x == host_low.x()
                  && device_offset.y == host_low.y()
                  && device_offset.z == host_low.z()
                  && device_size.x   == host_size.x()
                  && device_size.y   == host_size.y()
                  && device_size.z   == host_size.z()) {
                proceedWithCopy = true;

                // Note, race condition possible here
                bool finalized = dw->isFinalized();
                if (finalized) {
                  dw->unfinalize();
                }
                if (uses_SHRT_MAX) {
                  gridVar->allocate(host_low, host_high);
                } else {
                  // DS 12132019: GPU Resize fix. Use max ghost cells
                  // and corresponding low and high to allocate
                  // scratch space

                  // DS 06162020 Fix for the crash. Using
                  // allocateAndPut deletes the existing variable in
                  // dw and allocates a new one. If some other thread
                  // is accessing the same variable at the same time,
                  // then it leads to a crash. Hence allocate only if
                  // the variable does not exist on the host. While
                  // reusing existing variable, call getGridVar and
                  // set exactWindow=1. exactWindow=1 ensures that the
                  // allocated space has exactly same size as the
                  // requested. This is needed for D2H copy.

                  // Check comments in OnDemandDW::allocateAndPut,
                  // OnDemandDW::getGridVar, Array3<T>::rewindowExact
                  // and UnifiedScheduler::initiateD2H

                  // TODO: Throwing error if allocated and requested
                  // spaces are not same might be a problem for
                  // RMCRT. Fix can be to create a temporary variable
                  // (buffer) in UnifiedScheduler for D2H copy and
                  // then copy from buffer to actual variable. But
                  // lets try this solution first.

                  if (!dw->exists(dependantVar->m_var, matlID, patch))
                    dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch, dependantVar->m_var->getMaxDeviceGhostType(), dependantVar->m_var->getMaxDeviceGhost());
                  else // if the variable exists, then fetch it.
                    dw->getGridVar(*gridVar, dependantVar->m_var, matlID, patch, dependantVar->m_var->getMaxDeviceGhostType(), 0, 1); // do not pass ghost cells. We dont want to gather them, just need memory allocated
                }
                if (finalized) {
                  dw->refinalize();
                }
              } else {
                // They didn't match.  Lets see if the device var
                // doesn't have ghost cells.  This can happen prior to
                // the first timestep during initial computations when
                // no variables had room for ghost cells.
                Patch::getGhostOffsets(type, Ghost::None, 0, host_lowOffset, host_highOffset);
                patch->computeExtents(basis, dependantVar->m_var->getBoundaryLayer(), host_lowOffset, host_highOffset, host_low, host_high);

                host_size = host_high - host_low;
                if (   device_offset.x == host_low.x()
                    && device_offset.y == host_low.y()
                    && device_offset.z == host_low.z()
                    && device_size.x   == host_size.x()
                    && device_size.y   == host_size.y()
                    && device_size.z   == host_size.z()) {

                  proceedWithCopy = true;

                  // Note, race condition possible here
                  bool finalized = dw->isFinalized();
                  if (finalized) {
                    dw->unfinalize();
                  }

                  if (!dw->exists(dependantVar->m_var, matlID, patch))
                    dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch, Ghost::None, 0);
                  else
                    dw->getGridVar(*gridVar, dependantVar->m_var, matlID, patch, Ghost::None, 0, 1);

                  if (finalized) {
                    dw->refinalize();
                  }
                } else {
                  // The sizes STILL don't match. One more last ditch
                  // effort.  Assume it was using up to 32768 ghost cells.
                  level->findCellIndexRange(host_low, host_high);
                  host_size = host_high - host_low;
                  if (device_offset.x == host_low.x()
                       && device_offset.y == host_low.y()
                       && device_offset.z == host_low.z()
                       && device_size.x == host_size.x()
                       && device_size.y == host_size.y()
                       && device_size.z == host_size.z()) {

                    // ok, this worked.  Allocate it the large ghost
                    // cell way with getRegion
                    // Note, race condition possible here
                    bool finalized = dw->isFinalized();
                    if (finalized) {
                      dw->unfinalize();
                    }
                    gridVar->allocate(host_low, host_high);
                    if (finalized) {
                      dw->refinalize();
                    }
                    proceedWithCopy = true;
                  } else {
                    printf("ERROR:\nDetailedTask::initiateD2H() - Device and host sizes didn't match.  Device size is (%d, %d, %d), and host size is (%d, %d, %d)\n", device_size.x, device_size.y, device_size.y,host_size.x(), host_size.y(),host_size.z());
                    SCI_THROW( InternalError("DetailedTask::initiateD2H() - Device and host sizes didn't match.", __FILE__, __LINE__));
                  }
                }
              }

              // if offset and size is equal to CPU DW, directly copy
              // back to CPU var memory;
              if (proceedWithCopy) {

                host_ptr = gridVar->getBasePointer();
                host_bytes = gridVar->getDataSize();

                if (host_bytes == 0) {
                  printf("ERROR:\nDetailedTask::initiateD2H() - Transfer bytes is listed as zero.\n");
                  SCI_THROW( InternalError("DetailedTask::initiateD2H() - Transfer bytes is listed as zero.", __FILE__, __LINE__));
                }
                if (!host_ptr) {
                  printf("ERROR:\nDetailedTask::initiateD2H() - Invalid host pointer, it was nullptr.\n");
                  SCI_THROW( InternalError("DetailedTask::initiateD2H() - Invalid host pointer, it was nullptr.", __FILE__, __LINE__));
                }

#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
                m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
                m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                          deviceNum,
                                          host_ptr, device_ptr,
                                          host_bytes, cudaMemcpyDeviceToHost);
#else
                cudaStream_t* stream = this->getCudaStreamForThisTask(deviceNum);
                // ARS cudaMemcpyAsync copy from device to host
                // Kokkos equivalent - deep copy.
                cudaError_t retVal;

                CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
#endif
                IntVector temp(0,0,0);
                this->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      false,
                                                      IntVector(device_size.x, device_size.y, device_size.z),
                                                      elementDataSize, host_bytes,
                                                      IntVector(device_offset.x, device_offset.y, device_offset.z),
                                                      dependantVar,
                                                      gtype, numGhostCells,  deviceNum,
                                                      gridVar, GpuUtilities::sameDeviceSameMpiRank);

                // ARS cuda error handling. If there is an error
                // not sure the code would get here as the
                // previous call to CUDA_RT_SAFE_CALL would exit
                // upon an error.
                // if (retVal == cudaErrorLaunchFailure) {
                //   SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ this->getName(), __FILE__, __LINE__));
                // } else {
                //   // Kokkos equivalent - not needed??
                //   CUDA_RT_SAFE_CALL(retVal);
                // }
              }
              // delete gridVar;
            }
            break;
          }
          case TypeDescription::PerPatch: {
            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {

              PerPatchBase* hostPerPatchVar = dynamic_cast<PerPatchBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const bool finalized = dw->isFinalized();
              if (finalized) {
                dw->unfinalize();
              }
              dw->put(*hostPerPatchVar, dependantVar->m_var, matlID, patch);
              if (finalized) {
                dw->refinalize();
              }
              host_ptr = hostPerPatchVar->getBasePointer();
              host_bytes = hostPerPatchVar->getDataSize();

              GPUPerPatchBase* gpuPerPatchVar = OnDemandDataWarehouse::createGPUPerPatch(datatype);
              gpudw->get(*gpuPerPatchVar, varName.c_str(), patchID, matlID, levelID);
              device_ptr = gpuPerPatchVar->getVoidPointer();
              size_t device_bytes = gpuPerPatchVar->getMemSize();
              delete gpuPerPatchVar;

              // TODO: Verify no memory leaks
              if (host_bytes == device_bytes) {
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
                m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
                m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                          deviceNum,
                                          host_ptr, device_ptr,
                                          host_bytes, cudaMemcpyDeviceToHost);

#else
                cudaStream_t* stream = this->getCudaStreamForThisTask(deviceNum);
                // ARS cudaMemcpyAsync copy from device to host
                // Kokkos equivalent - deep copy.
                CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
#endif
                this->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      host_bytes, host_bytes,
                                                      dependantVar,
                                                      deviceNum,
                                                      hostPerPatchVar,
                                                      GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("ERROR: InitiateD2H - PerPatch variable memory sizes didn't match\n");
                SCI_THROW(InternalError("InitiateD2H - PerPatch variable memory sizes didn't match", __FILE__, __LINE__));
              }
              // delete hostPerPatchVar;
            }

            break;
          }
          case TypeDescription::ReductionVariable: {
            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {
              ReductionVariableBase* hostReductionVar = dynamic_cast<ReductionVariableBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const bool finalized = dw->isFinalized();
              if (finalized) {
                dw->unfinalize();
              }
              dw->put(*hostReductionVar, dependantVar->m_var, patch->getLevel(), matlID);
              if (finalized) {
                dw->refinalize();
              }
              host_ptr   = hostReductionVar->getBasePointer();
              host_bytes = hostReductionVar->getDataSize();

              GPUReductionVariableBase* gpuReductionVar = OnDemandDataWarehouse::createGPUReductionVariable(datatype);
              gpudw->get(*gpuReductionVar, varName.c_str(), patchID, matlID, levelID);
              device_ptr = gpuReductionVar->getVoidPointer();
              size_t device_bytes = gpuReductionVar->getMemSize();
              delete gpuReductionVar;

              if (host_bytes == device_bytes) {
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
                m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
                m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                          deviceNum,
                                          host_ptr, device_ptr,
                                          host_bytes, cudaMemcpyDeviceToHost);
#else
                cudaStream_t* stream = this->getCudaStreamForThisTask(deviceNum);
                // ARS cudaMemcpyAsync copy from device to host
                // Kokkos equivalent - deep copy.
                CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
#endif
                this->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      host_bytes, host_bytes,
                                                      dependantVar,
                                                      deviceNum,
                                                      hostReductionVar,
                                                      GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("ERROR: InitiateD2H - Reduction variable memory sizes didn't match\n");
                SCI_THROW(InternalError("InitiateD2H - Reduction variable memory sizes didn't match", __FILE__, __LINE__));
              }
              // delete hostReductionVar;
            }
            break;
          }
          default: {
            cerrLock.lock();
            {
              std::cerr << "Variable " << varName << " is of a type that is not supported on GPUs yet." << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
  }
}

// ______________________________________________________________________
//
void
DetailedTask::createTaskGpuDWs()
{
  // Create GPU datawarehouses for this specific task only.  They will
  // get copied into the GPU.  This is sizing these datawarehouses
  // dynamically and doing it all in only one alloc per datawarehouse.
  // See the bottom of the GPUDataWarehouse.h for more information.
#ifdef TASK_MANAGES_EXECSPACE
  deviceNumSet m_deviceNums = m_task->getDeviceNums(reinterpret_cast<intptr_t>(this));
#endif
  for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;
    unsigned int numItemsInDW = this->getTaskVars().getTotalVars(currentDevice, Task::OldDW) + this->getGhostVars().getNumGhostCellCopies(currentDevice, Task::OldDW);
    if (numItemsInDW > 0) {

      size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
          - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
          + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;

      GPUDataWarehouse* old_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
      // cudaHostRegister(old_taskGpuDW, objectSizeInBytes, cudaHostRegisterDefault);
      std::ostringstream out;
      out << "Old task GPU DW" << " MPIRank: " << Uintah::Parallel::getMPIRank() << " Task: " << this->getTask()->getName();
      old_taskGpuDW->init(currentDevice, out.str());
      old_taskGpuDW->setDebug(false);

      old_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);
      this->setTaskGpuDataWarehouse(currentDevice, Task::OldDW, old_taskGpuDW);
    }

    numItemsInDW = this->getTaskVars().getTotalVars(currentDevice, Task::NewDW) + this->getGhostVars().getNumGhostCellCopies(currentDevice, Task::NewDW);
    if (numItemsInDW > 0) {

      size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
          - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
          + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;
      GPUDataWarehouse* new_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
      // cudaHostRegister(new_taskGpuDW, objectSizeInBytes, cudaHostRegisterDefault);
      std::ostringstream out;
      out << "New task GPU DW"
          << " MPIRank: " << Uintah::Parallel::getMPIRank()
//        << " Thread:" << Impl::t_tid // ARS - FIXME
          << " Task: " << this->getName();
      new_taskGpuDW->init(currentDevice, out.str());
      new_taskGpuDW->setDebug(false);
      new_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);

      this->setTaskGpuDataWarehouse(currentDevice, Task::NewDW, new_taskGpuDW);
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::assignDevicesAndStreams()
{
  // Figure out which device this patch was assigned to.  If a task
  // has multiple patches, then assign all.  Most tasks should only
  // end up on one device.  Only tasks like data archiver's output
  // variables work on multiple patches which can be on multiple
  // devices.
  std::map<const Patch *, int>::iterator it;
  for (int p = 0; p < this->getPatches()->size(); p++) {
    const Patch* patch = this->getPatches()->get(p);
    int index = GpuUtilities::getGpuIndexForPatch(patch);
    if (index >= 0) {
      // See if this task doesn't yet have a stream for this GPU device.
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
      m_task->assignDevicesAndInstances(reinterpret_cast<intptr_t>(this));
#else
      m_task->assignDevicesAndStreams(reinterpret_cast<intptr_t>(this));
#endif
#else
      for (int i = 0; i < this->getTask()->maxStreamsPerTask(); i++) {
        if (this->getCudaStreamForThisTask(i) == nullptr) {
          this->assignDevice(0);
          cudaStream_t* stream = GPUMemoryPool::getCudaStreamFromPool(this, i);
          this->setCudaStreamForThisTask(i, stream);
        }
      }
#endif
    } else {
      cerrLock.lock();
      {
        std::cerr << "ERROR: Could not find the assigned GPU for this patch." << std::endl;
      }
      cerrLock.unlock();
      exit(-1);
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::assignDevicesAndStreamsFromGhostVars()
{
  // Go through the ghostVars collection and look at the patch where
  // all ghost cells are going.
  deviceNumSet & destinationDevices = this->getGhostVars().getDestinationDevices();

  for( auto &deviceNum : destinationDevices) {
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
    m_task->assignDevicesAndInstances(reinterpret_cast<intptr_t>(this), deviceNum);
#else
    m_task->assignDevicesAndStreams(reinterpret_cast<intptr_t>(this), deviceNum);
#endif
#else
    // See if this task was already assigned a stream.
    if (this->getCudaStreamForThisTask(deviceNum) == nullptr) {
      this->assignDevice(deviceNum);
      cudaStream_t* stream = GPUMemoryPool::getCudaStreamFromPool(this, deviceNum);
      this->setCudaStreamForThisTask(deviceNum, stream);
    }
#endif
  }
}

// ______________________________________________________________________
//
void
DetailedTask::assignStatusFlagsToPrepareACpuTask()
{
  // Keep track of all variables created or modified by a CPU task.
  // It also keeps track of ghost cells for a task.  This method seems
  // more like fitting a square peg into a round hole.  It tries to
  // temporarily bridge a gap between the OnDemand Data Warehouse and
  // the GPU Data Warehouse.  The OnDemand DW allocates variables on
  // the fly during task execution and also inflates vars to gather
  // ghost cells on task execution.  The GPU DW prepares all variables
  // and manages ghost cell copies prior to task execution.

  // This method was designed to solve a use case where a CPU task
  // created a var, then another CPU task modified it, then a GPU task
  // required it, then a CPU output task needed it.  Because the CPU
  // variable didn't get status flags attached to it due to it being
  // in the OnDemand Data Warehouse, the Kokkos Scheduler assumed the
  // only copy of the variable existed in GPU memory so it copied it
  // out of GPU memory into host memory right in the middle of when
  // the CPU output task was executing, causing a concurrency race
  // condition because that variable was already in host memory.  By
  // trying to track the host memory statuses for variables, this
  // should hopefully prevent those race conditions.

  // This probably isn't perfect, but should get us through the next
  // few months, and hopefully gets replaced when we can remove the
  // "OnDemand" part of the OnDemand Data Warehouse with a Kokkos
  // DataWarehouse.

  // Loop through all computes.  Create status flags of "allocating"
  // for them.  Do not track ghost cells, as ghost cells are created by
  // copying a

  // Loop through all modifies.  Create status flags of "allocated",
  // undoing any "valid" flags.

  // Loop through all requires.  If they have a ghost cell
  // requirement, we can't do much about it.
}


// ______________________________________________________________________
//
void
DetailedTask::findIntAndExtGpuDependencies(std::vector<OnDemandDataWarehouseP> & m_dws
                                           , std::set<std::string> &m_no_copy_data_vars
                                           , const VarLabel * m_reloc_new_pos_label
                                           , const VarLabel * m_parent_reloc_new_pos_label
                                           , int iteration
                                           , int t_id
                                             )
{
  this->clearPreparationCollections();

  // Prepare internal dependencies.  Only makes sense if we have multiple GPUs that we are using.
  if (Uintah::Parallel::usingDevice()) {

    // Prepare external dependencies.  The only thing that needs to be
    // prepared is getting ghost cell data from a GPU into a flat
    // array and copied to host memory so that the MPI engine can
    // treat it normally.  That means this handles GPU->other node GPU
    // and GPU->other node CPU.
    //

    for (DependencyBatch* batch = this->getComputes(); batch != 0;
        batch = batch->m_comp_next) {
      for (DetailedDep* req = batch->m_head; req != 0; req = req->m_next) {
        if ((req->m_comm_condition == DetailedDep::FirstIteration && iteration > 0) || (req->m_comm_condition == DetailedDep::SubsequentIterations && iteration == 0)
            || (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          continue;
        }

        // if we send/recv to an output task, don't send/recv if not an output timestep

        // ARS NOTE: Outputing and Checkpointing may be done out of
        // snyc now. I.e. turned on just before it happens rather than
        // turned on before the task graph execution.  As such, one
        // should also be checking:

        // m_application->activeReductionVariable( "outputInterval" );
        // m_application->activeReductionVariable( "checkpointInterval" );

        // However, if active the code below would be called regardless
        // if an output or checkpoint time step or not. Not sure that is
        // desired but not sure of the effect of not calling it and doing
        // an out of sync output or checkpoint.
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output &&
            // !m_output->isOutputTimeStep() &&
            // !m_output->isCheckpointTimeStep()) {
            !m_task_group->getSchedulerCommon()->getOutput()->isOutputTimeStep() &&
            !m_task_group->getSchedulerCommon()->getOutput()->isCheckpointTimeStep()) {
          continue;
        }
        OnDemandDataWarehouse* dw = m_dws[req->m_req->mapDataWarehouse()].get_rep();

        const VarLabel* posLabel;
        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in
        // the old dw on the prev timestep - pass it in if the
        // particle data is on the old dw

        // if (!m_reloc_new_pos_label && m_parent_scheduler) {
        if (!m_reloc_new_pos_label && m_parent_reloc_new_pos_label) {
          posDW    = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
          // posLabel = m_parent_scheduler->m_reloc_new_pos_label;
          posLabel = m_parent_reloc_new_pos_label;
        }
        else {
          // on an output task (and only on one) we require particle
          // variables from the NewDW
          if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)].get_rep();
          }
          else {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)].get_rep();
          }
          posLabel = m_reloc_new_pos_label;
        }
        // Load information which will be used to later invoke a
        // kernel to copy this range out of the GPU.
        prepareGpuDependencies(batch, posLabel, dw, posDW, req, GpuUtilities::anotherMpiRank);
      }
    }  // end for (DependencyBatch * batch = task->getComputes() )
  }
}


// ______________________________________________________________________
//
void
DetailedTask::syncTaskGpuDWs()
{
#ifdef TASK_MANAGES_EXECSPACE
  // For each GPU datawarehouse, see if there are ghost cells listed
  // to be copied if so, launch a kernel that copies them.
  GPUDataWarehouse *taskgpudw;
  deviceNumSet m_deviceNums = m_task->getDeviceNums(reinterpret_cast<intptr_t>(this));
  for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice,Task::OldDW);
    if (taskgpudw) {
      m_task->syncTaskGpuDW(reinterpret_cast<intptr_t>(this),
                            currentDevice, taskgpudw);
    }

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice,Task::NewDW);
    if (taskgpudw) {
      m_task->syncTaskGpuDW(reinterpret_cast<intptr_t>(this),
                            currentDevice, taskgpudw);
    }
  }
#else
  // For each GPU datawarehouse, see if there are ghost cells listed
  // to be copied if so, launch a kernel that copies them.
  GPUDataWarehouse *taskgpudw;
  for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;
    cudaStream_t* stream = this->getCudaStreamForThisTask(currentDevice);

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice,Task::OldDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(stream);
    }

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice,Task::NewDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(stream);
    }
  }
#endif
}


// ______________________________________________________________________
//
void
DetailedTask::performInternalGhostCellCopies()
{
#ifdef TASK_MANAGES_EXECSPACE
  // For each GPU datawarehouse, see if there are ghost cells listed
  // to be copied if so, launch a kernel that copies them.
  GPUDataWarehouse *taskgpudw;
  deviceNumSet m_deviceNums = m_task->getDeviceNums(reinterpret_cast<intptr_t>(this));

  for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice, Task::OldDW);

    if (taskgpudw != nullptr && taskgpudw->ghostCellCopiesNeeded()) {
      m_task->copyGpuGhostCellsToGpuVars(reinterpret_cast<intptr_t>(this),
                                         currentDevice, taskgpudw);
    }

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice, Task::NewDW);

    if (taskgpudw != nullptr && taskgpudw->ghostCellCopiesNeeded()) {
      m_task->copyGpuGhostCellsToGpuVars(reinterpret_cast<intptr_t>(this),
                                         currentDevice, taskgpudw);
    }
  }
#else
  // For each GPU datawarehouse, see if there are ghost cells listed
  // to be copied if so, launch a kernel that copies them.
  GPUDataWarehouse *taskgpudw;
  for (deviceNumSetIter deviceNums_it = m_deviceNums.begin(); deviceNums_it != m_deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;

    cudaStream_t* stream = this->getCudaStreamForThisTask(currentDevice);

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice, Task::OldDW);

    if (taskgpudw != nullptr && taskgpudw->ghostCellCopiesNeeded()) {
      taskgpudw->copyGpuGhostCellsToGpuVarsInvoker(stream);
    }

    taskgpudw = this->getTaskGpuDataWarehouse(currentDevice, Task::NewDW);

    if (taskgpudw != nullptr && taskgpudw->ghostCellCopiesNeeded()) {
      taskgpudw->copyGpuGhostCellsToGpuVarsInvoker(stream);
    }
  }
#endif
}


// ______________________________________________________________________
//
void
DetailedTask::copyAllGpuToGpuDependences(std::vector<OnDemandDataWarehouseP> & m_dws)
{
  // Iterate through the ghostVars, find all whose destination is
  // another GPU same MPI rank Get the destination device, the size
  // and do a straight GPU to GPU copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = this->getGhostVars().getMap();

  for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    printf( "copyAllGpuToGpuDependences num ghost vars \n");
    if (it->second.m_dest == GpuUtilities::anotherDeviceSameMpiRank) {
      // TODO: Needs a particle section

      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
      int3 device_source_offset;
      int3 device_source_size;

      // Get the source variable from the source GPU DW
      void *device_source_ptr;
      size_t elementDataSize = it->second.m_xstride;
      size_t memSize = ghostSize.x() * ghostSize.y() * ghostSize.z() * elementDataSize;
      GPUGridVariableBase* device_source_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
      GPUDataWarehouse* gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
      gpudw->getStagingVar(*device_source_var,
                 it->first.m_label.c_str(),
                 it->second.m_sourcePatchPointer->getID(),
                 it->first.m_matlIndx,
                 it->first.m_levelIndx,
                 make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                 make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_source_var->getArray3(device_source_offset, device_source_size, device_source_ptr);

      // Get the destination variable from the destination GPU DW
      gpudw = dw->getGPUDW(it->second.m_destDeviceNum);
      int3 device_dest_offset;
      int3 device_dest_size;
      void *device_dest_ptr;
      GPUGridVariableBase* device_dest_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      gpudw->getStagingVar(*device_dest_var,
                     it->first.m_label.c_str(),
                     it->second.m_destPatchPointer->getID(),
                     it->first.m_matlIndx,
                     it->first.m_levelIndx,
                     make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                     make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_dest_var->getArray3(device_dest_offset, device_dest_size, device_dest_ptr);

      // We can run peer copies from the source or the device stream.
      // While running it from the device technically is said to be a
      // bit slower, it's likely just to an extra event being created
      // to manage blocking the destination stream.  By putting it on
      // the device we are able to not need a synchronize step after
      // all the copies, because any upcoming API call will use the
      // streams and be naturally queued anyway.  When a copy
      // completes, anything placed in the destination stream can then
      // process.

      // Note: If we move to UVA, then we could just do a straight memcpy

      // Base call is commented out
      // OnDemandDataWarehouse::uintahSetCudaDevice(it->second.m_destDeviceNum);

#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
      m_task->doKokkosMemcpyPeerAsync(reinterpret_cast<intptr_t>(this),
#else
      m_task->doCudaMemcpyPeerAsync(reinterpret_cast<intptr_t>(this),
#endif
                                    it->second.m_destDeviceNum,
                                    device_dest_ptr,   it->second.m_destDeviceNum,
                                    device_source_ptr, it->second.m_sourceDeviceNum,
                            memSize);
#else
      cudaStream_t* stream = this->getCudaStreamForThisTask(it->second.m_destDeviceNum);
      // ARS cudaMemcpyPeerAsync copy from device to device
      // Kokkos equivalent - ??
      CUDA_RT_SAFE_CALL(cudaMemcpyPeerAsync(device_dest_ptr, it->second.m_destDeviceNum, device_source_ptr, it->second.m_sourceDeviceNum, memSize, *stream));
#endif
    }
  }
}


// ______________________________________________________________________
//
void
DetailedTask::copyAllExtGpuDependenciesToHost(std::vector<OnDemandDataWarehouseP> & m_dws)
{

  bool copiesExist = false;

  // If we put it in ghostVars, then we copied it to an array on the
  // GPU (D2D).  Go through the ones that indicate they are going to
  // another MPI rank.  Copy them out to the host (D2H).  To make the
  // engine cleaner for now, we'll then do a H2H copy step into the
  // variable.

  // In the future, to be more efficient, we could skip the host to
  // host copy and instead have sendMPI() send the array we get from
  // the device instead.

  // To be even more efficient than that, if everything is pinned,
  // unified addressing set up, and CUDA aware MPI used, then we could
  // pull everything out via MPI that way and avoid the manual D2H
  // copy and the H2H copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = this->getGhostVars().getMap();
  for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    // TODO: Needs a particle section
    if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
      void* host_ptr    = nullptr;    // host base pointer to raw data
      void* device_ptr  = nullptr;    // device base pointer to raw data
      size_t host_bytes = 0;
      IntVector host_low, host_high, host_offset, host_size, host_strides;
      int3 device_offset;
      int3 device_size;


      // We created a temporary host variable for this earlier, and
      // the deviceVars collection knows about it.  It's set as a
      // foreign var.
      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
      DeviceGridVariableInfo item = this->getDeviceVars().getStagingItem(it->first.m_label,
                 it->second.m_sourcePatchPointer,
                 it->first.m_matlIndx,
                 it->first.m_levelIndx,
                 ghostLow,
                 ghostSize,
                 it->first.m_dataWarehouse);
      GridVariableBase* tempGhostVar = (GridVariableBase*)item.m_var;

      tempGhostVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);

      host_ptr = tempGhostVar->getBasePointer();
      host_bytes = tempGhostVar->getDataSize();

      // copy the computes data back to the host
      // d2hComputesLock_.writeLock();
      // {

        GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
        OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
        GPUDataWarehouse* gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
        gpudw->getStagingVar(*device_var,
                   it->first.m_label.c_str(),
                   it->second.m_sourcePatchPointer->getID(),
                   it->first.m_matlIndx,
                   it->first.m_levelIndx,
                   make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                   make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
        device_var->getArray3(device_offset, device_size, device_ptr);

        // if offset and size is equal to CPU DW, directly copy back
        // to CPU var memory;
        if (device_offset.x == host_low.x()
            && device_offset.y == host_low.y()
            && device_offset.z == host_low.z()
            && device_size.x == host_size.x()
            && device_size.y == host_size.y()
            && device_size.z == host_size.z()) {

          // Base call is commented out
          // OnDemandDataWarehouse::uintahSetCudaDevice(it->second.m_sourceDeviceNum);

          // Since we know we need a stream, obtain one.
#ifdef TASK_MANAGES_EXECSPACE
#ifdef USE_KOKKOS_INSTANCE
          m_task->doKokkosDeepCopy(reinterpret_cast<intptr_t>(this),
#else
          m_task->doCudaMemcpyAsync(reinterpret_cast<intptr_t>(this),
#endif
                                    it->second.m_sourceDeviceNum,
                                    host_ptr, device_ptr,
                                    host_bytes, cudaMemcpyDeviceToHost);
#else
          cudaStream_t* stream = this->getCudaStreamForThisTask(it->second.m_sourceDeviceNum);
          // ARS cudaMemcpyAsync copy from host to device - replace with
          // Kokkos equivalent - deep copy.
          CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
#endif
          copiesExist = true;
        } else {
          std::cerr << "unifiedSCheduler::GpuDependenciesToHost() - Error - The host and device variable sizes did not match.  Cannot copy D2H." << std::endl;
          SCI_THROW(InternalError("Error - The host and device variable sizes did not match.  Cannot copy D2H", __FILE__, __LINE__));
        }

      // }
      // d2hComputesLock_.writeUnlock();

      delete device_var;
    }
  }

  if (copiesExist) {

    // Wait until all streams are done.  Further optimization could be
    // to check each stream one by one and make copies before waiting
    // for other streams to complete.
    // TODO: There's got to be a better way to do this.

    // ARS - FIXME This should replaced with a Kokkos::fence??
    // Kokkos::fence("copyAllExtGpuDependenciesToHost "
    //            "waiting for copies to finish");
#ifdef USE_KOKKOS_INSTANCE
    while (!this->checkAllKokkosInstancesDoneForThisTask()) {
#else
    while (!this->checkAllCudaStreamsDoneForThisTask()) {
#endif
      // printf("Sleeping\n");
    }

    for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {

      if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
        // TODO: Needs a particle section
        IntVector host_low, host_high, host_offset, host_size, host_strides;
        OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];

        // We created a temporary host variable for this earlier,
        // and the deviceVars collection knows about it.
        IntVector ghostLow = it->first.m_sharedLowCoordinates;
        IntVector ghostHigh = it->first.m_sharedHighCoordinates;
        IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
        DeviceGridVariableInfo item = this->getDeviceVars().getStagingItem(it->first.m_label, it->second.m_sourcePatchPointer,
                                                                            it->first.m_matlIndx, it->first.m_levelIndx, ghostLow,
                                                                            ghostSize, it->first.m_dataWarehouse);
        GridVariableBase* tempGhostVar = (GridVariableBase*)item.m_var;

        // Also get the existing host copy
        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(it->second.m_label->typeDescription()->createInstance());

        // Get the coordinate low/high of the host copy.
        const Patch * patch = it->second.m_sourcePatchPointer;
        TypeDescription::Type type = it->second.m_label->typeDescription()->getType();
        IntVector lowIndex, highIndex;
        bool uses_SHRT_MAX = (item.m_dep->m_num_ghost_cells == SHRT_MAX);
        if (uses_SHRT_MAX) {
          const Level * level = patch->getLevel();
          level->computeVariableExtents(type, lowIndex, highIndex);
        } else {
          Patch::VariableBasis basis = Patch::translateTypeToBasis(it->second.m_label->typeDescription()->getType(), false);
          patch->computeVariableExtents(basis, item.m_dep->m_var->getBoundaryLayer(), item.m_dep->m_gtype, item.m_dep->m_num_ghost_cells, lowIndex, highIndex);
        }

        // If it doesn't exist yet on the host, create it.  If it does
        // exist on the host, then if we got here that meant the host
        // data was invalid and the device data was valid, so nuke the
        // old contents and create a new one.  (Should we just get a
        // mutable var instead as it should be the size we already
        // need?)  This process is admittedly a bit hacky, as now the
        // var will be both partially valid and invalid.  The ghost
        // cell region is now valid on the host, while the rest of the
        // host var would be invalid.  Since we are writing to an old
        // data warehouse (from device to host), we need to
        // temporarily unfinalize it.
        const bool finalized = dw->isFinalized();
        if (finalized) {
          dw->unfinalize();
        }

        if (!dw->exists(item.m_dep->m_var, it->first.m_matlIndx, it->second.m_sourcePatchPointer)) {
          dw->allocateAndPut(*gridVar, item.m_dep->m_var, it->first.m_matlIndx,
                             it->second.m_sourcePatchPointer, item.m_dep->m_gtype,
                             item.m_dep->m_num_ghost_cells);
        } else {
          // Get a const variable in a non-constant way.
          // This assumes the variable has already been resized properly, which is why ghost cells are set to zero.
          // TODO: Check sizes anyway just to be safe.
          dw->getModifiable(*gridVar, item.m_dep->m_var, it->first.m_matlIndx, it->second.m_sourcePatchPointer, Ghost::None, 0);

        }
        // Do a host-to-host copy to bring the device data now on the host into the host-side variable so
        // that sendMPI can easily find the data as if no GPU were involved at all.
        gridVar->copyPatch(tempGhostVar, ghostLow, ghostHigh );
        if(finalized) {
          dw->refinalize();
        }

        // let go of our reference counters.
        delete gridVar;
        delete tempGhostVar;
      }
    }
  }
}

#endif // end defined(HAVE_GPU)

// ______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string
DetailedTask::myRankThread()
{
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank()
   // << "." << Impl::t_tid // ARS - FIXME
      ;
  return out.str();
}
