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

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/DependencyBatch.h>
#include <CCA/Components/Schedulers/MemoryLog.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUMemoryPool.h>
#endif

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Time.h>

#include <mutex>

#include <sci_defs/config_defs.h>
#include <sci_defs/cuda_defs.h>

namespace {

// These are for uniquely identifying the Uintah::CrowdMonitors<Tag>
// used to protect multi-threaded access to global data structures
struct external_ready_tag{};
struct internal_ready_tag{};

using  external_ready_monitor = Uintah::CrowdMonitor<external_ready_tag>;
using  internal_ready_monitor = Uintah::CrowdMonitor<internal_ready_tag>;

std::mutex internal_dependency_mutex{};

#ifdef HAVE_CUDA
  struct device_transfer_complete_queue_tag{};
  struct device_finalize_prep_queue_tag{};
  struct device_ready_queue_tag{};
  struct device_completed_queue_tag{};
  struct host_finalize_prep_queue_tag{};
  struct host_ready_queue_tag{};

  using  device_transfer_complete_queue_monitor = Uintah::CrowdMonitor<device_transfer_complete_queue_tag>;
  using  device_finalize_prep_queue_monitor     = Uintah::CrowdMonitor<device_finalize_prep_queue_tag>;
  using  device_ready_queue_monitor             = Uintah::CrowdMonitor<device_ready_queue_tag>;
  using  device_completed_queue_monitor         = Uintah::CrowdMonitor<device_completed_queue_tag>;
  using  host_finalize_prep_queue_monitor       = Uintah::CrowdMonitor<host_finalize_prep_queue_tag>;
  using  host_ready_queue_monitor               = Uintah::CrowdMonitor<host_ready_queue_tag>;
#endif

}

using namespace Uintah;
using namespace std;

// sync cout/cerr so they are readable when output by multiple threads
extern std::mutex cerrLock;
extern std::mutex coutLock;

extern Dout g_mpi_dbg;

static DebugStream dbg(         "DetailedTasks", false);
static DebugStream scrubout(    "Scrubbing",     false);
static DebugStream messagedbg(  "MessageTags",   false);
static DebugStream internaldbg( "InternalDeps",  false);
static DebugStream dwdbg(       "DetailedDWDBG", false);

// for debugging - set the var name to watch one in the scrubout
static std::string dbgScrubVar   = "";
static int         dbgScrubPatch = -1;

//_____________________________________________________________________________
//
DetailedTasks::DetailedTasks(       SchedulerCommon* sc,
                              const ProcessorGroup*  pg,
                                    DetailedTasks*   first,
                              const TaskGraph*       taskgraph,
                              const set<int>&        neighborhood_processors,
                                    bool             mustConsiderInternalDependencies /* = false */ ) :
  sc_(sc),
  d_myworld(pg),
  first(first),
  taskgraph_(taskgraph),
  mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
  currentDependencyGeneration_(1),
  extraCommunication_(0)
{
  // Set up mappings for the initial send tasks
  int dwmap[Task::TotalDWs];
  for (int i = 0; i < Task::TotalDWs; i++) {
    dwmap[i] = Task::InvalidDW;
  }
  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = Task::NoDW;

  stask_ = scinew Task( "send old data", Task::InitialSend );
  stask_->m_phase = 0;
  stask_->setMapping( dwmap );

  // Create a send old detailed task for every processor in my neighborhood.
  for (std::set<int>::iterator iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); iter++) {
    DetailedTask* newtask = scinew DetailedTask( stask_, 0, 0, this );
    newtask->assignResource(*iter);

    //use a map because the processors in this map are likely to be sparse
    sendoldmap_[*iter] = tasks_.size();
    tasks_.push_back(newtask);
  }
}

//_____________________________________________________________________________
//
DetailedTasks::~DetailedTasks()
{
  // Free dynamically allocated SrubItems
  (first ? first->scrubCountTable_ : scrubCountTable_).remove_all();

  for (int i = 0; i < (int)batches_.size(); i++) {
    delete batches_[i];
  }

  for (int i = 0; i < (int)tasks_.size(); i++) {
    delete tasks_[i];
  }

  delete stask_;
}

//_____________________________________________________________________________
//
void
DetailedTasks::assignMessageTags( int me )
{
  // maps from, to (process) pairs to indices for each batch of that pair
  std::map<std::pair<int, int>, int> perPairBatchIndices;

  for (int i = 0; i < (int)batches_.size(); i++) {
    DependencyBatch* batch = batches_[i];
    int from = batch->m_from_task->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, d_myworld->size());
    int to = batch->m_to_rank;
    ASSERTRANGE(to, 0, d_myworld->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      std::pair<int, int> fromToPair = std::make_pair(from, to);
      batches_[i]->m_message_tag = ++perPairBatchIndices[fromToPair];  // start with one
      if (messagedbg.active()) {
        coutLock.lock();
        messagedbg << "Rank-" << me << " assigning message tag " << batch->m_message_tag << " from task " << batch->m_from_task->getName()
                   << " to task " << batch->m_to_tasks.front()->getName() << ", rank-" << from << " to rank-" << to << "\n";
        coutLock.unlock();
      }
    }
  }

  if (dbg.active()) {
    std::map<std::pair<int, int>, int>::iterator iter;
    for (iter = perPairBatchIndices.begin(); iter != perPairBatchIndices.end(); iter++) {
      int from = iter->first.first;
      int to = iter->first.second;
      int num = iter->second;
      dbg << num << " messages from rank-" << from << " to rank-" << to << "\n";
    }
  }
}  // end assignMessageTags()

//_____________________________________________________________________________
//
void
DetailedTasks::add( DetailedTask* task )
{
  tasks_.push_back(task);
}

//_____________________________________________________________________________
//
void
DetailedTasks::makeDWKeyDatabase()
{
  for (int i = 0; i < (int)localtasks_.size(); i++) {
    DetailedTask* task = localtasks_[i];
    //for reduction task check modifies other task check computes
    const Task::Dependency *comp = task->getTask()->isReductionTask() ? task->getTask()->getModifies() : task->getTask()->getComputes();
    for (; comp != 0; comp = comp->m_next) {
      const MaterialSubset* matls = comp->m_matls ? comp->m_matls : task->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable ||
            comp->m_var->typeDescription()->getType() == TypeDescription::SoleVariable) {
          levelKeyDB.insert(comp->m_var, matl, comp->m_reduction_level);
        }
        else { // if variables saved on varDB
          const PatchSubset* patches = comp->m_patches ? comp->m_patches : task->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            varKeyDB.insert(comp->m_var, matl, patch);
            if (dwdbg.active()) {
              dwdbg << "reserve " << comp->m_var->getName() << " on Patch " << patch->getID() << ", Matl " << matl << "\n";
            }
          }
        }
      }  //end matls
    } // end comps
  } // end localtasks
}

//_____________________________________________________________________________
//
void
DetailedTasks::computeLocalTasks( int me )
{
  if (localtasks_.size() != 0) {
    return;
  }

  int order = 0;
  initiallyReadyTasks_ = TaskQueue();
  for (int i = 0; i < (int)tasks_.size(); i++) {
    DetailedTask* task = tasks_[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
    if (task->getAssignedResourceIndex() == me || task->getTask()->getType() == Task::Reduction) {
      localtasks_.push_back(task);

      if (task->areInternalDependenciesSatisfied()) {
        initiallyReadyTasks_.push(task);
      }
      task->assignStaticOrder(++order);
    }
  }
}

//_____________________________________________________________________________
//
DetailedTask::DetailedTask(       Task*           task,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DetailedTasks*  taskGroup )
  :   task(task),
      patches(patches),
      matls(matls),
      comp_head(0),
      internal_comp_head(0),
      taskGroup(taskGroup),
      numPendingInternalDependencies(0),
      resourceIndex(-1),
      staticOrder(-1),
      d_profileType(Normal)
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
  if (patches && patches->removeReference()) {
    delete patches;
  }
  if (matls && matls->removeReference()) {
    delete matls;
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::doit( const ProcessorGroup*                 pg,
                          vector<OnDemandDataWarehouseP>& oddws,
                          vector<DataWarehouseP>&         dws,
                          Task::CallBackEvent             event /* = Task::CPU */ )
{
  if (internaldbg.active()) {
    cerrLock.lock();
    internaldbg << "DetailedTask " << this << " begin doit()\n";
    internaldbg << " task is " << task << "\n";
    internaldbg << "   num Pending Deps: " << numPendingInternalDependencies << "\n";
    internaldbg << "   Originally needed deps (" << internalDependencies.size() << "):\n";

    std::list<InternalDependency>::iterator iter = internalDependencies.begin();

    for (int i = 0; iter != internalDependencies.end(); iter++, i++) {
      internaldbg << i << ":    " << *((*iter).prerequisiteTask->getTask()) << "\n";
    }
    cerrLock.unlock();
  }
  for (int i = 0; i < (int)dws.size(); i++) {
    if (oddws[i] != 0) {
      oddws[i]->pushRunningTask(task, &oddws);
    }
  }

#ifdef HAVE_CUDA
  // determine if task will be executed on CPU or device, e.g. GPU or MIC
  if (task->usesDevice()) {
    //Run the GPU task.  Technically the engine has structure to run one task on multiple devices if
    //that task had patches on multiple devices.  So run the task once per device.  As often as possible,
    //we want to design tasks so each task runs on only once device, instead of a one to many relationship.
    for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums_.begin(); deviceNums_it != deviceNums_.end(); ++deviceNums_it) {
      const unsigned int currentDevice = *deviceNums_it;
      OnDemandDataWarehouse::uintahSetCudaDevice(currentDevice);
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
      task->doit(this, event, pg, patches, matls, dws,
                 device_oldtaskdw,
                 device_newtaskdw,
                 getCudaStreamForThisTask(currentDevice), currentDevice);
    }
  }
  else {
    task->doit(this, event, pg, patches, matls, dws, nullptr, nullptr, nullptr, -1);
  }
#else
  task->doit(this, event, pg, patches, matls, dws, nullptr, nullptr, nullptr, -1);
#endif

  for (int i = 0; i < (int)dws.size(); i++) {
    if (oddws[i] != 0) {
      oddws[i]->checkTasksAccesses(patches, matls);
      oddws[i]->popRunningTask();
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::initializeScrubs( vector<OnDemandDataWarehouseP>& dws,
                                 int                             dwmap[] )
{
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " Begin initialize scrubs\n";
  }

  vector<bool> initialized(dws.size(), false);
  for (int i = 0; i < (int)Task::TotalDWs; i++) {
    if (dwmap[i] < 0) {
      continue;
    }
    OnDemandDataWarehouse* dw = dws[dwmap[i]].get_rep();
    // TODO APH - clean this up (06/09/16)
//    if (dw != 0) dw->copyKeyDB(varKeyDB, levelKeyDB);
    if (dw != 0 && dw->getScrubMode() == DataWarehouse::ScrubComplete) {
      // only a OldDW or a CoarseOldDW will have scrubComplete 
      //   But we know a future taskgraph (in a w-cycle) will need the vars if there are fine dws 
      //   between New and Old.  In this case, the scrub count needs to be complemented with CoarseOldDW
      int tgtype = getTaskGraph()->getType();
      if (!initialized[dwmap[i]] || tgtype == Scheduler::IntermediateTaskGraph) {
        // if we're intermediate, we're going to need to make sure we don't scrub CoarseOld before we finish using it
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << dw->getID() << " for DW type " << i << " ADD="
                 << initialized[dwmap[i]] << '\n';
        dw->initializeScrubs(i, &(first ? first->scrubCountTable_ : scrubCountTable_), initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD="
                 << 1 << '\n';
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &(first ? first->scrubCountTable_ : scrubCountTable_), true);
      }
      initialized[dwmap[i]] = true;
    }
  }
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " End initialize scrubs\n";
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::scrub( std::vector<OnDemandDataWarehouseP>& dws )
{
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " Starting scrub after task: " << *this << '\n';
  }
  const Task* task = getTask();

  const std::set<const VarLabel*, VarLabel::Compare>& initialRequires = taskGroup->getSchedulerCommon()->getInitialRequiredVars();
  const std::set<std::string>& unscrubbables = taskGroup->getSchedulerCommon()->getNoScrubVars();

  // Decrement the scrub count for each of the required variables
  for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->m_next) {
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

          for (int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];

            if (patch->getLevel()->getIndex() > 0 && patch != neighbor && req->m_patches_dom == Task::ThisLevel) {
              // don't scrub on AMR overlapping patches...
              IntVector l = low, h = high;
              l = Max(neighbor->getExtraLowIndex(basis, req->m_var->getBoundaryLayer()), low);
              h = Min(neighbor->getExtraHighIndex(basis, req->m_var->getBoundaryLayer()), high);
              patch->cullIntersection(basis, req->m_var->getBoundaryLayer(), neighbor->getRealPatch(), l, h);
              if (l == h)
                continue;
            }
            if (req->m_patches_dom == Task::FineLevel) {
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
                count = dws[dw]->decrementScrubCount(req->m_var, matls->get(m), neighbor);
                if (scrubout.active() && (req->m_var->getName() == dbgScrubVar || dbgScrubVar == "")
                    && (neighbor->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                  scrubout << Parallel::getMPIRank() << "   decrementing scrub count for requires of " << dws[dw]->getID() << "/"
                           << neighbor->getID() << "/" << matls->get(m) << "/" << req->m_var->getName() << ": " << count
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
  for (const Task::Dependency* mod = task->getModifies(); mod != 0; mod = mod->m_next) {
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
            if (scrubout.active() && (mod->m_var->getName() == dbgScrubVar || dbgScrubVar == "")
                && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1))
              scrubout << Parallel::getMPIRank() << "   decrementing scrub count for modifies of " << dws[dw]->getID() << "/"
              << patch->getID() << "/" << matls->get(m) << "/" << mod->m_var->getName() << ": " << count
              << (count == 0 ? " - scrubbed\n" : "\n");
          }
        }
      }
    }
  }

  // Set the scrub count for each of the computes variables
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
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
            if (taskGroup->getScrubCount(comp->m_var, matl, patch, whichdw, count)) {
              if (scrubout.active() && (comp->m_var->getName() == dbgScrubVar || dbgScrubVar == "")
                  && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                scrubout << Parallel::getMPIRank() << "   setting scrub count for computes of " << dws[dw]->getID() << "/"
                         << patch->getID() << "/" << matls->get(m) << "/" << comp->m_var->getName() << ": " << count << '\n';
              }
              dws[dw]->setScrubCount(comp->m_var, matl, patch, count);
            }
            else {
              // Not in the scrub map, must be never needed...
              if (scrubout.active() && (comp->m_var->getName() == dbgScrubVar || dbgScrubVar == "")
                  && (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1)) {
                scrubout << Parallel::getMPIRank() << "   trashing variable immediately after compute: " << dws[dw]->getID() << "/"
                         << patch->getID() << "/" << matls->get(m) << "/" << comp->m_var->getName() << '\n';
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
// used to be in terms of the dw index within the scheduler,
// but now store WhichDW.  This enables multiple tg execution
void
DetailedTasks::addScrubCount( const VarLabel* var,
                                    int       matlindex,
                              const Patch*    patch,
                                    int       dw )
{
  if (patch->isVirtual())
    patch = patch->getRealPatch();
  ScrubItem key(var, matlindex, patch, dw);
  ScrubItem* result;
  result = (first ? first->scrubCountTable_ : scrubCountTable_).lookup(&key);
  if (!result) {
    result = scinew ScrubItem(var, matlindex, patch, dw);
    (first ? first->scrubCountTable_ : scrubCountTable_).insert(result);
  }
  result->m_count++;
  if (scrubout.active() && (var->getName() == dbgScrubVar || dbgScrubVar == "")
      && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1)) {
    scrubout << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/" << patch->getID() << "/" << matlindex
             << "/" << *var << ": " << result->m_count << std::endl;
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::setScrubCount( const Task::Dependency*                    req,
                                    int                                  matl,
                              const Patch*                               patch,
                                    vector<OnDemandDataWarehouseP>&      dws )
{
  ASSERT(!patch->isVirtual());
  DataWarehouse::ScrubMode scrubmode = dws[req->mapDataWarehouse()]->getScrubMode();
  const std::set<const VarLabel*, VarLabel::Compare>& initialRequires = getSchedulerCommon()->getInitialRequiredVars();
  if (scrubmode == DataWarehouse::ScrubComplete || (scrubmode == DataWarehouse::ScrubNonPermanent
      && initialRequires.find(req->m_var) == initialRequires.end())) {
    int scrubcount;
    if (!getScrubCount(req->m_var, matl, patch, req->m_whichdw, scrubcount)) {
      SCI_THROW(InternalError("No scrub count for received MPIVariable: "+req->m_var->getName(), __FILE__, __LINE__));
    }

    if (scrubout.active() && (req->m_var->getName() == dbgScrubVar || dbgScrubVar == "")
        && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1)) {
      scrubout << Parallel::getMPIRank() << " setting scrubcount for recv of " << req->mapDataWarehouse() << "/" << patch->getID()
               << "/" << matl << "/" << req->m_var->getName() << ": " << scrubcount << '\n';
    }
    dws[req->mapDataWarehouse()]->setScrubCount(req->m_var, matl, patch, scrubcount);
  }
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getScrubCount( const VarLabel* label,
                                    int       matlIndex,
                              const Patch*    patch,
                                    int       dw,
                                    int&      count )
{
  ASSERT(!patch->isVirtual());
  ScrubItem key(label, matlIndex, patch, dw);
  ScrubItem* result = (first ? first->scrubCountTable_ : scrubCountTable_).lookup(&key);
  if (result) {
    count = result->m_count;
    return true;
  }
  else {
    return false;
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::createScrubCounts()
{
  // Clear old ScrubItems
  (first ? first->scrubCountTable_ : scrubCountTable_).remove_all();

  // Go through each of the tasks and determine which variables it will require
  for (int i = 0; i < (int)localtasks_.size(); i++) {
    DetailedTask* dtask = localtasks_[i];
    const Task* task = dtask->getTask();
    for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->m_next) {
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->m_whichdw;
      TypeDescription::Type type = req->m_var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          Patch::selectType neighbors;
          IntVector low, high;
          patch->computeVariableExtents(type, req->m_var->getBoundaryLayer(), req->m_gtype, req->m_num_ghost_cells, neighbors, low, high);
          for (int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];
            for (int m = 0; m < matls->size(); m++) {
              addScrubCount(req->m_var, matls->get(m), neighbor, whichdw);
            }
          }
        }
      }
    }

    // determine which variables this task will modify
    for (const Task::Dependency* req = task->getModifies(); req != 0; req = req->m_next) {
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->m_whichdw;
      TypeDescription::Type type = req->m_var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            addScrubCount(req->m_var, matls->get(m), patch, whichdw);
          }
        }
      }
    }
  }
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " scrub counts:\n";
    scrubout << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for (FastHashTableIter<ScrubItem> iter(&(first ? first->scrubCountTable_ : scrubCountTable_)); iter.ok(); ++iter) {
      const ScrubItem* rec = iter.get_key();
      scrubout << rec->m_dw << '/' << (rec->m_patch ? rec->m_patch->getID() : 0) << '/' << rec->m_matl << '/' << rec->m_label->getName()
               << "\t\t" << rec->m_count << '\n';
    }
    scrubout << Parallel::getMPIRank() << " end scrub counts\n";
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::findRequiringTasks( const VarLabel*            var,
                                        list<DetailedTask*>& requiringTasks )
{
  // find requiring tasks

  // find external requires
  for (DependencyBatch* batch = getComputes(); batch != 0; batch = batch->m_comp_next) {
    for (DetailedDep* dep = batch->m_head; dep != 0; dep = dep->m_next) {
      if (dep->m_req->m_var == var) {
        requiringTasks.insert(requiringTasks.end(), dep->m_to_tasks.begin(), dep->m_to_tasks.end());
      }
    }
  }

  // find internal requires
  std::map<DetailedTask*, InternalDependency*>::iterator internalDepIter;
  for (internalDepIter = internalDependents.begin(); internalDepIter != internalDependents.end(); ++internalDepIter) {
    if (internalDepIter->second->vars.find(var) != internalDepIter->second->vars.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

//_____________________________________________________________________________
//
DetailedDep*
DetailedTasks::findMatchingDetailedDep(       DependencyBatch*  batch,
                                              DetailedTask*     toTask,
                                              Task::Dependency* req,
                                        const Patch*            fromPatch,
                                              int               matl,
                                              IntVector         low,
                                              IntVector         high,
                                              IntVector&        totalLow,
                                              IntVector&        totalHigh,
                                              DetailedDep*&     parent_dep )
{
  totalLow = low;
  totalHigh = high;
  DetailedDep* dep = batch->m_head;

  parent_dep = 0;
  DetailedDep* last_dep = 0;
  DetailedDep* valid_dep = 0;

  //search each dep
  for (; dep != 0; dep = dep->m_next) {
    //if deps are equivalent
    if (fromPatch == dep->m_from_patch && matl == dep->m_matl
        && (req == dep->m_req || (req->m_var->equals(dep->m_req->m_var) && req->mapDataWarehouse() == dep->m_req->mapDataWarehouse()))) {

      // total range - the same var in each dep needs to have the same patchlow/high
      dep->m_patch_low = totalLow = Min(totalLow, dep->m_patch_low);
      dep->m_patch_high = totalHigh = Max(totalHigh, dep->m_patch_high);

      int ngcDiff = req->m_num_ghost_cells > dep->m_req->m_num_ghost_cells ? (req->m_num_ghost_cells - dep->m_req->m_num_ghost_cells) : 0;
      IntVector new_l = Min(low, dep->m_low);
      IntVector new_h = Max(high, dep->m_high);
      IntVector newRange = new_h - new_l;
      IntVector requiredRange = high - low;
      IntVector oldRange = dep->m_high - dep->m_low + IntVector(ngcDiff, ngcDiff, ngcDiff);
      int newSize = newRange.x() * newRange.y() * newRange.z();
      int requiredSize = requiredRange.x() * requiredRange.y() * requiredRange.z();
      int oldSize = oldRange.x() * oldRange.y() * oldRange.z();

      bool extraComm = newSize > requiredSize + oldSize;

      if (sc_->useSmallMessages()) {
        // If two patches on the same processor want data from the same patch on a different
        // processor, we can either pack them in one dependency and send the min and max of their range (which
        // will frequently result in sending the entire patch), or we can use two dependencies (which will get packed into
        // one message) and only send the resulting data.

        // We want to create a new dep in such cases.  However, we don't want to do this in cases where we simply add more
        // ghost cells.
        if (!extraComm) {
          //combining does not create extra communication so take possibly this dep;
          //first check if the dep completely includes this dep
          if (dep->m_low == new_l && dep->m_high == new_h) {
            //take this dep
            parent_dep = last_dep;
            valid_dep = dep;
            break;
          }
          else {
            //only take the dep if we haven't already found one
            if (valid_dep == 0) {
              parent_dep = last_dep;
              valid_dep = dep;
            }
            //keep searching in case there is a better dep to combine with
          }
        }
        else if (dbg.active()) {
          dbg << d_myworld->myrank() << "            Ignoring: " << dep->m_low << " " << dep->m_high << ", fromPatch = ";
          if (fromPatch) {
            dbg << fromPatch->getID() << '\n';
          }
          else {
            dbg << "nullptr\n";
          }
          dbg << d_myworld->myrank() << " TP: " << totalLow << " " << totalHigh << std::endl;
        }
      }
      else {
        if (extraComm) {
          extraCommunication_ += newSize - (requiredSize + oldSize);
        }
        //not using small messages so take the first dep you find, it will be extended
        valid_dep = dep;
        break;
      }
    }
    //pointer to dependency before this dep so insertion/deletion can be done quicker
    last_dep = dep;
  }
  if (valid_dep == 0) {
    parent_dep = last_dep;
  }

  return valid_dep;
}

/*************************
 * This function will create the detailed dependency for the
 * parameters passed in.  If a similar detailed dependency
 * already exists it will combine those dependencies into a single
 * dependency.  
 *
 * Dependencies are ordered from oldest to newest in a linked list.  It is vital that 
 * this order is maintained.  Failure to maintain this order can cause messages to be combined 
 * inconsistently across different tasks causing various problems.  New dependencies are added
 * to the end of the list.  If a dependency was combined then the extended dependency is added
 * at the same location that i was first combined.  This is to ensure all future dependencies
 * combine with the same dependencies as the original.
 */
void
DetailedTasks::possiblyCreateDependency(       DetailedTask*              from,
                                               Task::Dependency*          comp,
                                         const Patch*                     fromPatch,
                                               DetailedTask*              to,
                                               Task::Dependency*          req,
                                         const Patch*                     toPatch,
                                               int                        matl,
                                         const IntVector&                 low,
                                         const IntVector&                 high,
                                               DetailedDep::CommCondition cond )
{
  ASSERTRANGE(from->getAssignedResourceIndex(), 0, d_myworld->size());
  ASSERTRANGE(to->getAssignedResourceIndex(),   0, d_myworld->size());

  if (dbg.active()) {
    cerrLock.lock();
    {
      dbg << d_myworld->myrank() << "          " << *to << " depends on " << *from << "\n";
      if (comp) {
        dbg << d_myworld->myrank() << "            From comp " << *comp;
      }
      else {
        dbg << d_myworld->myrank() << "            From OldDW ";
      }
      dbg << " to req " << *req << '\n';
    }
    cerrLock.unlock();
  }

  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();

  // if neither task talks to this processor, return
  if (fromresource != d_myworld->myrank() && toresource != d_myworld->myrank()) {
    return;
  }

  if ((toresource == d_myworld->myrank() || (req->m_patches_dom != Task::ThisLevel && fromresource == d_myworld->myrank()))
      && fromPatch && !req->m_var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->m_var, matl, fromPatch, req->m_whichdw);
  }

  //if the dependency is on the same processor then add an internal dependency
  if (fromresource == d_myworld->myrank() && fromresource == toresource) {
    to->addInternalDependency(from, req->m_var);

    //In case of multiple GPUs per node, we don't return.  Multiple GPUs
    //need internal dependencies to communicate data.
    if ( ! Uintah::Parallel::usingDevice()) {
      return;
    }

  }

  //this should have been pruned out earlier
  ASSERT(!req->m_var->typeDescription()->isReductionVariable())
  // Do not check external deps on SoleVariable
  if (req->m_var->typeDescription()->getType() == TypeDescription::SoleVariable) {
    return;
  }
#ifdef HAVE_CUDA
  if (Uintah::Parallel::usingDevice()) {
    if (fromresource == d_myworld->myrank() && fromresource == toresource) {
      if (fromPatch != toPatch) {
        //printf("In DetailedTasks::createInternalDependencyBatch creating internal dependency from patch %d to patch %d, from task %p to task %p\n", fromPatch->getID(), toPatch->getID(), from, to);
        createInternalDependencyBatch(from, comp, fromPatch, to, req, toPatch, matl, low, high, cond);
      }
      return; //We've got all internal dependency information for the GPU, now we can return.
    }
  }
#endif

  //make keys for MPI messages
  if (fromPatch) varKeyDB.insert(req->m_var,matl,fromPatch);

  //get dependency batch
  DependencyBatch* batch = from->getComputes();

  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource) {
      break;
    }
  }

  //if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, to);
    batches_.push_back(batch);
    from->addComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addRequires(batch);
#else
    to->addRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "          NEW BATCH!\n";
    }
  }
  else if (mustConsiderInternalDependencies_) {  // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add to the batch's toTasks.
      batch->m_to_tasks.push_back(to);
    }
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
    }
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  // create the new dependency
  DetailedDep* new_dep = scinew DetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  // search for a dependency that can be combined with this dependency

  // location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high, varRangeLow,
                                                      varRangeHigh, parent_dep);

  // This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDep* insert_dep = parent_dep;

  // if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  while (matching_dep != 0) {

    //debugging output
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
          << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      dbg << *req->m_var << '\n';
      dbg << *new_dep->m_req->m_var << '\n';
      if (comp) {
        dbg << *comp->m_var << '\n';
      }
      if (new_dep->m_comp) {
        dbg << *new_dep->m_comp->m_var << '\n';
      }
    }

    // extend the dependency range
    new_dep->m_low = Min(new_dep->m_low, matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);

    // TODO APH - figure this out and clean up (01/31/16)
    /*
     //if the same dependency already exists then short circuit out of this function.
    if (matching_dep->low == new_dep->low && matching_dep->high == new_dep->high) {
      matching_dep->toTasks.splice(matching_dep->toTasks.begin(), new_dep->toTasks);
      delete new_dep;
      return;
    }
     */

    // copy matching dependencies toTasks to the new dependency
    new_dep->m_to_tasks.splice(new_dep->m_to_tasks.begin(), matching_dep->m_to_tasks);

    // erase particle sends/recvs
    if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x") {
        dbg << d_myworld->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
            << " cond " << cond << " dw " << req->mapDataWarehouse() << "\n";
      }

      if (fromresource == d_myworld->myrank()) {
        set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
        ASSERT(iter != particleSends_[toresource].end());

        //subtract one from the count
        iter->count_--;

        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          particleSends_[toresource].erase(iter);
        }
      }
      else if (toresource == d_myworld->myrank()) {
        set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
        ASSERT(iter != particleRecvs_[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          particleRecvs_[fromresource].erase(iter);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == nullptr) {
      batch->m_head = matching_dep->m_next;
    }
    else {
      parent_dep->m_next = matching_dep->m_next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high,
                                           varRangeLow, varRangeHigh, parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep) {
      insert_dep = parent_dep;
    }
  }

  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->m_patch_low = varRangeLow;
  new_dep->m_patch_high = varRangeHigh;

  if (insert_dep == nullptr) {
    //no dependencies are in the list so add it to the head
    batch->m_head = new_dep;
    new_dep->m_next = nullptr;
  }
  else {
    //dependencies already exist so add it at the insert location.
    new_dep->m_next = insert_dep->m_next;
    insert_dep->m_next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == d_myworld->myrank()) {
      set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
      if (iter == particleSends_[toresource].end()) {  //if does not exist
        //add to the sends list
        particleSends_[toresource].insert(pmg);
      }
      else {
        //increment count
        iter->count_++;
      }
    }
    else if (toresource == d_myworld->myrank()) {
      set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
      if (iter == particleRecvs_[fromresource].end()) {
        //add to the recvs list
        particleRecvs_[fromresource].insert(pmg);
      }
      else {
        //increment the count
        iter->count_++;
      }

    }
    if (req->m_var->getName() == "p.x") {
      dbg << d_myworld->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
          << req->mapDataWarehouse() << "\n";
    }
  }

  if (dbg.active()) {
    dbg << d_myworld->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch) {
      dbg << fromPatch->getID() << '\n';
    }
    else {
      dbg << "nullptr\n";
    }
  }
}

#ifdef HAVE_CUDA

void DetailedTasks::createInternalDependencyBatch(DetailedTask* from,
    Task::Dependency* comp,
    const Patch* fromPatch,
    DetailedTask* to,
    Task::Dependency* req,
    const Patch *toPatch,
    int matl,
    const IntVector& low,
    const IntVector& high,
    DetailedDep::CommCondition cond) {

  //get dependency batch
  DependencyBatch* batch = from->getInternalComputes();
  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();
  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource)
      break;
  }

  //if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, to);
    batches_.push_back(batch);  //Should be fine to push this batch on here, at worst
                                //MPI message tags are created for these which won't get used.
    from->addInternalComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addInternalRequires(batch);
#else
    to->addInternalRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active())
      dbg << d_myworld->myrank() << "          NEW BATCH!\n";
  } else if (mustConsiderInternalDependencies_) {  // i.e. threaded mode
    if (to->addInternalRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->m_to_tasks.push_back(to);
    }
    if (dbg.active())
      dbg << d_myworld->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  //create the new dependency
  DetailedDep* new_dep = scinew DetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high, varRangeLow,
                                                      varRangeHigh, parent_dep);

  //This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDep* insert_dep = parent_dep;
  //If two dependencies are going to be on two different GPUs,
  //then do not merge the two dependencies into one collection.
  //Instead, keep them separate.  If they will be on the same GPU
  //then we can merge the dependencies as normal.
  //This means that no


  //if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  //check if we sohuld combine them.  If the  new_dep->low
  while (matching_dep != 0) {

    //debugging output
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
          << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      dbg << *req->m_var << '\n';
      dbg << *new_dep->m_req->m_var << '\n';
      if (comp)
        dbg << *comp->m_var << '\n';
      if (new_dep->m_comp)
        dbg << *new_dep->m_comp->m_var << '\n';
    }



    //extend the dependency range
    new_dep->m_low = Min(new_dep->m_low, matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);

    //copy matching dependencies toTasks to the new dependency
    new_dep->m_to_tasks.splice(new_dep->m_to_tasks.begin(), matching_dep->m_to_tasks);

    //erase particle sends/recvs
    if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x")
        dbg << d_myworld->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
            << " cond " << cond << " dw " << req->mapDataWarehouse() << std::endl;

      if (fromresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
        ASSERT(iter!=particleSends_[toresource].end());

        //subtract one from the count
        iter->count_--;

        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          particleSends_[toresource].erase(iter);
        }
      } else if (toresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
        ASSERT(iter!=particleRecvs_[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          particleRecvs_[fromresource].erase(iter);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == nullptr) {
      batch->m_head = matching_dep->m_next;
    } else {
      parent_dep->m_next = matching_dep->m_next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high, varRangeLow, varRangeHigh,
                                           parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep)
      insert_dep = parent_dep;
  }



  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->m_patch_low = varRangeLow;
  new_dep->m_patch_high = varRangeHigh;


  if (insert_dep == nullptr) {
    // no dependencies are in the list so add it to the head
    batch->m_head = new_dep;
    new_dep->m_next = nullptr;
  } else {
    // dependencies already exist so add it at the insert location.
    new_dep->m_next = insert_dep->m_next;
    insert_dep->m_next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == d_myworld->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
      if (iter == particleSends_[toresource].end())  //if does not exist
          {
        //add to the sends list
        particleSends_[toresource].insert(pmg);
      } else {
        //increment count
        iter->count_++;
      }
    } else if (toresource == d_myworld->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
      if (iter == particleRecvs_[fromresource].end()) {
        //add to the recvs list
        particleRecvs_[fromresource].insert(pmg);
      } else {
        //increment the count
        iter->count_++;
      }

    }
    if (req->m_var->getName() == "p.x")
      dbg << d_myworld->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
          << req->mapDataWarehouse() << std::endl;
  }

  if (dbg.active()) {
    dbg << d_myworld->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch)
      dbg << fromPatch->getID() << '\n';
    else
      dbg << "nullptr\n";
  }

}

DetailedDep* DetailedTasks::findMatchingInternalDetailedDep(DependencyBatch* batch,
                                                    DetailedTask* toTask,
                                                    Task::Dependency* req,
                                                    const Patch* fromPatch,
                                                    int matl,
                                                    IntVector low,
                                                    IntVector high,
                                                    IntVector& totalLow,
                                                    IntVector& totalHigh,
                                                    DetailedDep* &parent_dep)
{
  totalLow = low;
  totalHigh = high;
  DetailedDep* dep = batch->m_head;

  parent_dep = 0;
  DetailedDep* last_dep = 0;
  DetailedDep* valid_dep = 0;
  //For now, turning off a feature that can combine ghost cells into larger vars
  //for scenarios where one source ghost cell var can handle more than one destination patch.

  //search each dep
  for (; dep != 0; dep = dep->m_next) {
    //Temporarily disable feature of merging source ghost cells in the same patch into
    //a larger var (so instead of two transfers, it can be done as one transfer)
    /*
    //if deps are equivalent
    if (fromPatch == dep->fromPatch && matl == dep->matl
        && (req == dep->req || (req->var->equals(dep->req->var) && req->mapDataWarehouse() == dep->req->mapDataWarehouse()))) {


      //For the GPUs, ensure that the destinations will be on the same device, and not another device
      //This assumes that a GPU task will not be assigned to multiple patches belonging to more than one device.
      if (getGpuIndexForPatch(toTask->getPatches()->get(0)) == getGpuIndexForPatch( dep->toTasks.front()->getPatches()->get(0))) {
        // total range - the same var in each dep needs to have the same patchlow/high
        dep->patchLow = totalLow = Min(totalLow, dep->patchLow);
        dep->patchHigh = totalHigh = Max(totalHigh, dep->patchHigh);

        int ngcDiff = req->numGhostCells > dep->req->numGhostCells ? (req->numGhostCells - dep->req->numGhostCells) : 0;
        IntVector new_l = Min(low, dep->low);
        IntVector new_h = Max(high, dep->high);
        IntVector newRange = new_h - new_l;
        IntVector requiredRange = high - low;
        IntVector oldRange = dep->high - dep->low + IntVector(ngcDiff, ngcDiff, ngcDiff);
        int newSize = newRange.x() * newRange.y() * newRange.z();
        int requiredSize = requiredRange.x() * requiredRange.y() * requiredRange.z();
        int oldSize = oldRange.x() * oldRange.y() * oldRange.z();

        //bool extraComm = newSize > requiredSize + oldSize;

        //if (sc_->useSmallMessages()) {
          // If two patches on the same processor want data from the same patch on a different
          // processor, we can either pack them in one dependency and send the min and max of their range (which
          // will frequently result in sending the entire patch), or we can use two dependencies (which will get packed into
          // one message) and only send the resulting data.

          // We want to create a new dep in such cases.  However, we don't want to do this in cases where we simply add more
          // ghost cells.
          //if (!extraComm) {
            //combining does not create extra communication so take possibly this dep;
            //first check if the dep completely includes this dep
            if (dep->low == new_l && dep->high == new_h) {
              //take this dep
              parent_dep = last_dep;
              valid_dep = dep;
              break;
            } else {
              //only take the dep if we haven't already found one
              if (valid_dep == 0) {
                parent_dep = last_dep;
                valid_dep = dep;
              }
              //keep searching in case there is a better dep to combine with
            }
        //  }
        //} else {
        //
        //  //not using small messages so take the first dep you find, it will be extended
        //  valid_dep = dep;
        //  break;
      }
    }

    */

    //pointer to dependency before this dep so insertion/deletion can be done quicker
    last_dep = dep;
  }

  if (valid_dep == 0)
    parent_dep = last_dep;

  return valid_dep;
}

#endif
//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getOldDWSendTask( int proc )
{
#if SCI_ASSERTION_LEVEL>0
  //verify the map entry has been created
  if (sendoldmap_.find(proc) == sendoldmap_.end()) {
    std::cout << d_myworld->myrank() << " Error trying to get oldDWSendTask for processor: " << proc << " but it does not exist\n";
    throw InternalError("oldDWSendTask does not exist", __FILE__, __LINE__);
  }
#endif 
  return tasks_[sendoldmap_[proc]];
}

//_____________________________________________________________________________
//
void
DetailedTask::addComputes( DependencyBatch* comp )
{
  comp->m_comp_next = comp_head;
  comp_head = comp;
}

//_____________________________________________________________________________
//
bool
DetailedTask::addRequires( DependencyBatch* req )
{
  // return true if it is adding a new batch
  return reqs.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
void DetailedTask::addInternalComputes(DependencyBatch* comp)
{
  comp->m_comp_next = internal_comp_head;
  internal_comp_head = comp;
}

//_____________________________________________________________________________
//
bool DetailedTask::addInternalRequires(DependencyBatch* req)
{
  // return true if it is adding a new batch
  return internal_reqs.insert(std::make_pair(req, req)).second;
}


//_____________________________________________________________________________
// can be called in one of two places - when the last MPI Recv has completed, or from MPIScheduler
void
DetailedTask::checkExternalDepCount()
{
  DOUT(g_mpi_dbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
               << " external deps: " << externalDependencyCount_.load(std::memory_order_seq_cst) << " internal deps: " << numPendingInternalDependencies);

  if (externalDependencyCount_.load(std::memory_order_seq_cst) == 0 && taskGroup->sc_->useInternalDeps() && initiated_ && !task->usesMPI()) {
    {
      external_ready_monitor external_ready_lock { Uintah::CrowdMonitor<external_ready_tag>::WRITER };
      DOUT(g_mpi_dbg, "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
                   << " MPI requirements satisfied, placing into external ready queue");

      if (externallyReady_ == false) {
        taskGroup->mpiCompletedTasks_.push(this);
        externallyReady_ = true;
      }
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::resetDependencyCounts()
{
  externalDependencyCount_.store(0, std::memory_order_seq_cst);
  externallyReady_ = false;
  initiated_       = false;
}

//_____________________________________________________________________________
//
void
DetailedTask::addInternalDependency(       DetailedTask* prerequisiteTask,
                                     const VarLabel*     var )
{
  if (taskGroup->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    map<DetailedTask*, InternalDependency*>::iterator foundIt = prerequisiteTask->internalDependents.find(this);
    if (foundIt == prerequisiteTask->internalDependents.end()) {
      internalDependencies.push_back(InternalDependency(prerequisiteTask, this, var, 0/* not satisfied */));
      prerequisiteTask->internalDependents[this] = &internalDependencies.back();
      numPendingInternalDependencies = internalDependencies.size();
      if (internaldbg.active()) {
        internaldbg << Parallel::getMPIRank() << " Adding dependency between " << *this << " and " << *prerequisiteTask << "\n";
      }

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


//For tasks where there are multiple devices for the task (i.e. data archiver output tasks)
std::set<unsigned int> DetailedTask::getDeviceNums() const
{
  return deviceNums_;
}


cudaStream_t* DetailedTask::getCudaStreamForThisTask(unsigned int deviceNum) const
{
  std::map <unsigned int, cudaStream_t*>::const_iterator it;
  it = d_cudaStreams.find(deviceNum);
  if (it != d_cudaStreams.end()) {
    return it->second;
  }
  return nullptr;
}


void DetailedTask::setCudaStreamForThisTask(unsigned int deviceNum, cudaStream_t* s)
{
  if (s == nullptr) {
    printf("ERROR! - DetailedTask::setCudaStreamForThisTask() - A request was made to assign a stream at address nullptr into this task %s\n", getName().c_str());
    SCI_THROW(InternalError("A request was made to assign a stream at address nullptr into this task :"+ getName() , __FILE__, __LINE__));
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


bool DetailedTask::checkCudaStreamDoneForThisTask(unsigned int deviceNum_) const
{

  // sets the CUDA context, for the call to cudaEventQuery()
  cudaError_t retVal;
  if (deviceNum_ != 0) {
    printf("Error, DetailedTask::checkCudaStreamDoneForThisTask is %u\n", deviceNum_);
    exit(-1);
  }
  OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum_);
  std::map<unsigned int, cudaStream_t*>::const_iterator it= d_cudaStreams.find(deviceNum_);
  if (it == d_cudaStreams.end()) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Request for stream information for device %d, but this task wasn't assigned any streams for this device.  For task %s\n", deviceNum_,  getName().c_str());
    SCI_THROW(InternalError("Request for stream information for a device, but it wasn't assigned any streams for that device.  For task: " + getName() , __FILE__, __LINE__));
    return false;
  }
  if (it->second == nullptr) {
    printf("ERROR! - DetailedTask::checkCudaStreamDoneForThisTask() - Stream pointer with nullptr address for task %s\n", getName().c_str());
    SCI_THROW(InternalError("Stream pointer with nullptr address for task: " + getName() , __FILE__, __LINE__));
    return false;
  }

  retVal = cudaStreamQuery(*(it->second));
  if (retVal == cudaSuccess) {
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
    temp.TaskGpuDW[0] = nullptr;
    temp.TaskGpuDW[1] = nullptr;
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
  return nullptr;

}

void DetailedTask::deleteTaskGpuDataWarehouses() {
  for (std::map<unsigned int, TaskGpuDataWarehouses>::iterator it = TaskGpuDWs.begin(); it != TaskGpuDWs.end(); ++it) {
    for (int i = 0; i < 2; i++) {
        if (it->second.TaskGpuDW[i] != nullptr) {
          //Note: Do not call the clear() method.  The Task GPU DWs only contains a "snapshot"
          //of the things in the GPU.  The host side GPU DWs is responsible for
          //deallocating all the GPU resources.  The only thing we do want to clean
          //up is that this GPUDW lives on the GPU.
          it->second.TaskGpuDW[i]->deleteSelfOnDevice();
          it->second.TaskGpuDW[i]->cleanup();

          free(it->second.TaskGpuDW[i]);
          it->second.TaskGpuDW[i] = nullptr;
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

void DetailedTask::addTempHostMemoryToBeFreedOnCompletion(void *ptr) {

  taskHostMemoryPoolItems.push(ptr);
}

void DetailedTask::addTempCudaMemoryToBeFreedOnCompletion(unsigned int device_id, void *ptr) {
  gpuMemoryPoolDevicePtrItem gpuItem(device_id, ptr);
  taskCudaMemoryPoolItems.push_back(gpuItem);
}

void DetailedTask::deleteTemporaryTaskVars() {

  //clean out the host list
  while (!taskHostMemoryPoolItems.empty()) {
    taskHostMemoryPoolItems.front();
    taskHostMemoryPoolItems.pop();
  }


  //and the device
  for (auto p : taskCudaMemoryPoolItems) {
    GPUMemoryPool::freeCudaSpaceFromPool(p.device_id, p.ptr);
  }
  taskCudaMemoryPoolItems.clear();

}

#endif // HAVE_CUDA


//_____________________________________________________________________________
//
void
DetailedTask::done( vector<OnDemandDataWarehouseP>& dws )
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  if (internaldbg.active()) {
    cerrLock.lock();
    internaldbg << "This: " << this << " is done with task: " << task << "\n";
    internaldbg << "Name is: " << task->getName() << " which has (" << internalDependents.size() << ") tasks waiting on it:\n";
    cerrLock.unlock();
  }

  int cnt = 1000;
  std::map<DetailedTask*, InternalDependency*>::iterator iter;
  for (iter = internalDependents.begin(); iter != internalDependents.end(); iter++) {
    InternalDependency* dep = (*iter).second;

    if (internaldbg.active()) {
      internaldbg << Parallel::getMPIRank() << " Dependency satisfied between " << *dep->dependentTask << " and " << *this << "\n";
    }

    dep->dependentTask->dependencySatisfied(dep);
    cnt++;
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::dependencySatisfied( InternalDependency* dep )
{
  internal_dependency_mutex.lock();
  {
    ASSERT(numPendingInternalDependencies > 0);
    unsigned long currentGeneration = taskGroup->getCurrentDependencyGeneration();

    // if false, then the dependency has already been satisfied
    ASSERT(dep->satisfiedGeneration < currentGeneration);

    dep->satisfiedGeneration = currentGeneration;
    numPendingInternalDependencies--;

    if (internaldbg.active()) {
      cerrLock.lock();
      internaldbg << *(dep->dependentTask->getTask()) << " has " << numPendingInternalDependencies << " left.\n";
      cerrLock.unlock();
    }

    if (internaldbg.active()) {
      internaldbg << Parallel::getMPIRank() << " satisfying dependency: prereq: " << *dep->prerequisiteTask << " dep: "
                  << *dep->dependentTask << " numPending: " << numPendingInternalDependencies << "\n";
    }

    if (numPendingInternalDependencies == 0) {
      taskGroup->internalDependenciesSatisfied(this);
      // reset for next timestep
      numPendingInternalDependencies = internalDependencies.size();
    }
  }
  internal_dependency_mutex.unlock();
}

namespace Uintah {

std::ostream&
operator<<(       std::ostream& out,
            const DetailedTask& task )
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

} // end namespace Uintah

//_____________________________________________________________________________
//
void
DetailedTasks::internalDependenciesSatisfied( DetailedTask* task )
{
  if (internaldbg.active()) {
    cerrLock.lock();
    internaldbg << "Begin internalDependenciesSatisfied\n";
    cerrLock.unlock();
  }

  {
    internal_ready_monitor internal_ready_lock{ Uintah::CrowdMonitor<internal_ready_tag>::WRITER };
    readyTasks_.push(task);
  }

  if (internaldbg.active()) {
    cerrLock.lock();
    internaldbg << *task << " satisfied.  Now " << readyTasks_.size() << " ready.\n";
    cerrLock.unlock();
  }
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  DetailedTask* nextTask = nullptr;
  {
    internal_ready_monitor internal_ready_lock{ Uintah::CrowdMonitor<internal_ready_tag>::WRITER };
    if (!readyTasks_.empty()) {
      nextTask = readyTasks_.front();
      readyTasks_.pop();
    }
  }
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numInternalReadyTasks()
{
  {
    internal_ready_monitor internal_ready_lock{ Uintah::CrowdMonitor<internal_ready_tag>::READER };
    return readyTasks_.size();
  }
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  DetailedTask* nextTask = nullptr;
  {
    external_ready_monitor external_ready_lock{ Uintah::CrowdMonitor<external_ready_tag>::WRITER };
    if (!mpiCompletedTasks_.empty()) {
      nextTask = mpiCompletedTasks_.top();
      mpiCompletedTasks_.pop();
    }
  }
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numExternalReadyTasks()
{
  {
    external_ready_monitor external_ready_lock{ Uintah::CrowdMonitor<external_ready_tag>::READER };
    return mpiCompletedTasks_.size();
  }
}

#ifdef HAVE_CUDA

//_____________________________________________________________________________
//
bool
DetailedTasks::getNextVerifyDataTransferCompletionTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    device_transfer_complete_queue_monitor transfer_queue_lock{ Uintah::CrowdMonitor<device_transfer_complete_queue_tag>::WRITER };
    if (!verifyDataTransferCompletionTasks_.empty()) {
      dtask = verifyDataTransferCompletionTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextVerifyDataTransferCompletionIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        verifyDataTransferCompletionTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}

//______________________________________________________________________
//
bool
DetailedTasks::getNextFinalizeDevicePreparationTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    device_finalize_prep_queue_monitor device_finalize_queue_lock{ Uintah::CrowdMonitor<device_finalize_prep_queue_tag>::WRITER };
    if (!finalizeDevicePreparationTasks_.empty()) {
      dtask = finalizeDevicePreparationTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextFinalizeDevicePreparationTaskIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        finalizeDevicePreparationTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getNextInitiallyReadyDeviceTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    device_ready_queue_monitor device_ready_queue_lock{ Uintah::CrowdMonitor<device_ready_queue_tag>::WRITER };
    if (!initiallyReadyDeviceTasks_.empty()) {
      dtask = initiallyReadyDeviceTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextInitiallyReadyDeviceTaskIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        initiallyReadyDeviceTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getNextCompletionPendingDeviceTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    device_completed_queue_monitor device_completed_queue_lock{ Uintah::CrowdMonitor<device_completed_queue_tag>::WRITER };
    if (!completionPendingDeviceTasks_.empty()) {
      dtask = completionPendingDeviceTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextCompletionPendingDeviceTaskIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        completionPendingDeviceTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getNextFinalizeHostPreparationTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    host_finalize_prep_queue_monitor host_finalize_queue_lock{ Uintah::CrowdMonitor<host_finalize_prep_queue_tag>::WRITER };
    if (!finalizeHostPreparationTasks_.empty()) {
      dtask = finalizeHostPreparationTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextFinalizeHostPreparationTaskIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        finalizeHostPreparationTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getNextInitiallyReadyHostTaskIfAble(DetailedTask* &dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;
  {
    host_ready_queue_monitor host_ready_queue_lock{ Uintah::CrowdMonitor<host_ready_queue_tag>::WRITER };
    if (!initiallyReadyHostTasks_.empty()) {
      dtask = initiallyReadyHostTasks_.front();
      if (!dtask) {
        SCI_THROW(InternalError("DetailedTasks::getNextInitiallyReadyHostTaskIfAble() - The task in this queue was a nullptr.  This shouldn't ever happen.", __FILE__, __LINE__));
      }
      if (dtask->checkAllCudaStreamsDoneForThisTask()) {
        initiallyReadyHostTasks_.pop();
        retVal = true;
      }
    }
    if (!retVal) {
      dtask = nullptr;
    }
  }

  return retVal;
}
//_____________________________________________________________________________
//
void DetailedTasks::addVerifyDataTransferCompletion(DetailedTask* dtask)
{
  {
    device_transfer_complete_queue_monitor transfer_queue_lock{ Uintah::CrowdMonitor<device_transfer_complete_queue_tag>::WRITER };
    verifyDataTransferCompletionTasks_.push(dtask);
  }
}

//_____________________________________________________________________________
//
void DetailedTasks::addFinalizeDevicePreparation(DetailedTask* dtask)
{
  {
    device_finalize_prep_queue_monitor device_finalize_queue_lock{ Uintah::CrowdMonitor<device_finalize_prep_queue_tag>::WRITER };
    if (!dtask) {
      SCI_THROW(InternalError("DetailedTasks::addFinalizeDevicePreparation() - Cannot add a nullptr to the queue.", __FILE__, __LINE__));
    }
    finalizeDevicePreparationTasks_.push(dtask);
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::addInitiallyReadyDeviceTask( DetailedTask* dtask )
{
  {
    device_ready_queue_monitor device_ready_queue_lock{ Uintah::CrowdMonitor<device_ready_queue_tag>::WRITER };
    initiallyReadyDeviceTasks_.push(dtask);
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::addCompletionPendingDeviceTask( DetailedTask* dtask )
{
  {
    device_completed_queue_monitor device_completed_queue_lock{ Uintah::CrowdMonitor<device_completed_queue_tag>::WRITER };
    completionPendingDeviceTasks_.push(dtask);
  }
}

//_____________________________________________________________________________
//
void DetailedTasks::addFinalizeHostPreparation(DetailedTask* dtask)
{
  {
    host_finalize_prep_queue_monitor host_finalize_queue_lock{ Uintah::CrowdMonitor<host_finalize_prep_queue_tag>::WRITER };
    finalizeHostPreparationTasks_.push(dtask);
  }
}

//_____________________________________________________________________________
//
void DetailedTasks::addInitiallyReadyHostTask(DetailedTask* dtask)
{
  {
    host_ready_queue_monitor host_ready_queue_lock{ Uintah::CrowdMonitor<host_ready_queue_tag>::WRITER };
    initiallyReadyHostTasks_.push(dtask);
  }
}

#endif


//_____________________________________________________________________________
//
void
DetailedTasks::initTimestep()
{
  readyTasks_ = initiallyReadyTasks_;
  incrementDependencyGeneration();
  initializeBatches();
}

//_____________________________________________________________________________
//
void
DetailedTasks::incrementDependencyGeneration()
{
  if (currentDependencyGeneration_ >= ULONG_MAX) {
    SCI_THROW(InternalError("DetailedTasks::currentDependencySatisfyingGeneration has overflowed", __FILE__, __LINE__));
  }
  currentDependencyGeneration_++;
}

//_____________________________________________________________________________
//
void
DetailedTasks::initializeBatches()
{
  for (int i = 0; i < (int)batches_.size(); i++) {
    batches_[i]->reset();
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::logMemoryUse(       ostream&       out,
                                   unsigned long& total,
                             const string&        tag )
{
  std::ostringstream elems1;
  elems1 << tasks_.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1, elems1.str(), tasks_.size() * sizeof(DetailedTask), 0);
  std::ostringstream elems2;
  elems2 << batches_.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1, elems2.str(), batches_.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)batches_.size(); i++) {
    for (DetailedDep* dep = batches_[i]->m_head; dep != 0; dep = dep->m_next) {
      ndeps++;
    }
  }
  std::ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "DetailedDep", 0, -1, elems3.str(), ndeps * sizeof(DetailedDep), 0);
}

//_____________________________________________________________________________
//
void
DetailedTasks::emitEdges( ProblemSpecP edgesElement,
                          int          rank )
{
  for (int i = 0; i < (int)tasks_.size(); i++) {
    ASSERTRANGE(tasks_[i]->getAssignedResourceIndex(), 0, d_myworld->size());
    if (tasks_[i]->getAssignedResourceIndex() == rank) {
      tasks_[i]->emitEdges(edgesElement);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::emitEdges( ProblemSpecP edgesElement )
{
  std::map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = reqs.begin(); req_iter != reqs.end(); req_iter++) {
    DetailedTask* fromTask = (*req_iter).first->m_from_task;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  std::list<InternalDependency>::iterator iter;
  for (iter = internalDependencies.begin(); iter != internalDependencies.end(); iter++) {
    DetailedTask* fromTask = (*iter).prerequisiteTask;
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

class PatchIDIterator {

  public:

    PatchIDIterator(const std::vector<const Patch*>::const_iterator& iter)
        : iter_(iter)
    {
    }

    PatchIDIterator& operator=(const PatchIDIterator& iter2)
    {
      iter_ = iter2.iter_;
      return *this;
    }

    int operator*()
    {
      const Patch* patch = *iter_;  //vector<Patch*>::iterator::operator*();
      return patch ? patch->getID() : -1;
    }

    PatchIDIterator& operator++()
    {
      iter_++;
      return *this;
    }

    bool operator!=(const PatchIDIterator& iter2)
    {
      return iter_ != iter2.iter_;
    }

  private:
    std::vector<const Patch*>::const_iterator iter_;
};

//_____________________________________________________________________________
//
std::string
DetailedTask::getName() const
{
  if (name_ != "") {
    return name_;
  }

  name_ = std::string(task->getName());

  if (patches != 0) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder(PatchIDIterator(patches->getVector().begin()), PatchIDIterator(patches->getVector().end()));
    name_ += std::string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if (matls != 0) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder(matls->getVector().begin(), matls->getVector().end());
    name_ += std::string(" (Matls: ") + matlSet.toString() + ")";
  }

  return name_;
}

//_____________________________________________________________________________
//
// comparing the priority of two detailed tasks True means give rtask priority
bool
DetailedTaskPriorityComparison::operator()( DetailedTask*& ltask,
                                            DetailedTask*& rtask )
{
  QueueAlg alg = ltask->getTaskGroup()->getTaskPriorityAlg();
  ASSERT(alg == rtask->getTaskGroup()->getTaskPriorityAlg());

  if (alg == FCFS) {
    return false;               //First Come First Serve;
  }

  if (alg == Stack) {
    return true;               //Fist Come Last Serve, for robust testing;
  }

  if (alg == Random) {
    return (random() % 2 == 0);   //Random;
  }

  if (ltask->getTask()->getSortedOrder() > rtask->getTask()->getSortedOrder()) {
    return true;
  }

  if (ltask->getTask()->getSortedOrder() < rtask->getTask()->getSortedOrder()) {
    return false;
  }

  if (alg == MostChildren) {
    return ltask->getTask()->m_child_tasks.size() < rtask->getTask()->m_child_tasks.size();
  }
  else if (alg == LeastChildren) {
    return ltask->getTask()->m_child_tasks.size() > rtask->getTask()->m_child_tasks.size();
  }
  else if (alg == MostAllChildren) {
    return ltask->getTask()->m_all_child_tasks.size() < rtask->getTask()->m_all_child_tasks.size();
  }
  else if (alg == LeastAllChildren) {
    return ltask->getTask()->m_all_child_tasks.size() > rtask->getTask()->m_all_child_tasks.size();
  }
  else if (alg == MostL2Children || alg == LeastL2Children) {
    int ll2 = 0;
    int rl2 = 0;
    set<Task*>::iterator it;
    for (it = ltask->getTask()->m_child_tasks.begin(); it != ltask->getTask()->m_child_tasks.end(); it++) {
      ll2 += (*it)->m_child_tasks.size();
    }
    for (it = rtask->getTask()->m_child_tasks.begin(); it != rtask->getTask()->m_child_tasks.end(); it++) {
      rl2 += (*it)->m_child_tasks.size();
    }
    if (alg == MostL2Children) {
      return ll2 < rl2;
    }
    else {
      return ll2 > rl2;
    }
  }
  else if (alg == MostMessages || alg == LeastMessages) {
    int lmsg = 0;
    int rmsg = 0;
    for (DependencyBatch* batch = ltask->getComputes(); batch != 0; batch = batch->m_comp_next) {
      for (DetailedDep* dep = batch->m_head; dep != 0; dep = dep->m_next) {
        lmsg++;
      }
    }
    for (DependencyBatch* batch = rtask->getComputes(); batch != 0; batch = batch->m_comp_next) {
      for (DetailedDep* dep = batch->m_head; dep != 0; dep = dep->m_next) {
        rmsg++;
      }
    }
    if (alg == MostMessages) {
      return lmsg < rmsg;
    }
    else {
      return lmsg > rmsg;
    }
  }
  else if (alg == PatchOrder) {  //smaller level, larger size, smaller patchied, smaller tasksortid
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    if (getLevel(lpatches) == getLevel(rpatches)) {
      if (lpatches->size() == rpatches->size()) {
        return lpatches->get(0)->getID() > rpatches->get(0)->getID();
      }
      else {
        return lpatches->size() < rpatches->size();
      }
    }
    else {
      return getLevel(lpatches) > getLevel(rpatches);
    }
  }
  else if (alg == PatchOrderRandom) {  //smaller level, larger size, smaller patchied, smaller tasksortid
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    if (getLevel(lpatches) == getLevel(rpatches)) {
      if (lpatches->size() == rpatches->size()) {
        return (random() % 2 == 0);
      }
      else {
        return lpatches->size() < rpatches->size();
      }
    }
    else {
      return getLevel(lpatches) > getLevel(rpatches);
    }
  }
  else {
    return false;
  }
}
