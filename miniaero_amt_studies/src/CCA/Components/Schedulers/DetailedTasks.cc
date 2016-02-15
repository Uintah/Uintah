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

#include <sci_defs/config_defs.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;

// sync cout/cerr so they are readable when output by multiple threads
extern SCIRun::Mutex cerrLock;
extern SCIRun::Mutex coutLock;

extern DebugStream mixedDebug;
extern DebugStream mpidbg;

static DebugStream dbg(         "DetailedTasks", false);
static DebugStream scrubout(    "Scrubbing",     false);
static DebugStream messagedbg(  "MessageTags",   false);
static DebugStream internaldbg( "InternalDeps",  false);
static DebugStream dwdbg(       "DetailedDWDBG", false);
static DebugStream waitout(     "WaitTimes",     false);

// for debugging - set the var name to watch one in the scrubout
static std::string dbgScrubVar   = "";
static int         dbgScrubPatch = -1;

std::map<std::string, double> DependencyBatch::s_wait_times{};

//_____________________________________________________________________________
//
DetailedTasks::DetailedTasks(       SchedulerCommon * sc
                            , const ProcessorGroup  * pg
                            ,       DetailedTasks   * first
                            , const TaskGraph       * taskgraph
                            , const std::set<int>   & neighborhood_processors
                            ,       bool              mustConsiderInternalDependencies /* = false */
                            )
  : m_sched{ sc }
  , m_proc_group{ pg }
  , m_first_task{ first }
  , m_task_graph{ taskgraph }
  , m_init_requires{ nullptr }
  , m_must_consider_internal_dependencies{ mustConsiderInternalDependencies }
  , m_task_priority_algorithm{ QueueAlg::MostMessages }
  , m_current_dependency_generation{ 1 }
  , m_extra_communication{ 0 }
#ifdef HAVE_CUDA
  ,
  deviceReadyQueueLock_("DetailedTasks Device Ready Queue"),
  deviceCompletedQueueLock_("DetailedTasks Device Completed Queue"),
  deviceFinalizePreparationQueueLock_("DetailedTasks Device Finalize Preparation Queue"),
  hostReadyQueueLock_("DetailedTasks Host Ready Queue")
#endif
{
  // Set up mappings for the initial send tasks
  int dwmap[Task::TotalDWs];
  for (int i = 0; i < Task::TotalDWs; i++) {
    dwmap[i] = Task::InvalidDW;
  }
  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = Task::NoDW;

  m_send_old_data_task = scinew Task("send old data", Task::InitialSend);
  m_send_old_data_task->d_phase = 0;
  m_send_old_data_task->setMapping(dwmap);

  // Create a send old detailed task for every processor in my neighborhood.
  for (std::set<int>::iterator iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); iter++) {
    DetailedTask* newtask = scinew DetailedTask(m_send_old_data_task, 0, 0, this);
    newtask->assignResource(*iter);
    //use a map because the processors in this map are likely to be sparse
    m_send_old_data_map[*iter] = m_tasks.size();
    m_tasks.push_back(newtask);
  }
}

//_____________________________________________________________________________
//
DetailedTasks::~DetailedTasks()
{
  // Free dynamically allocated SrubItems
  (m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count).remove_all();

  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    delete m_dep_batches[i];
  }

  for (int i = 0; i < (int)m_tasks.size(); i++) {
    delete m_tasks[i];
  }

  delete m_send_old_data_task;
}

//_____________________________________________________________________________
//
DependencyBatch::~DependencyBatch()
{
  DetailedDependency* dep = m_head;
  while (dep) {
    DetailedDependency* tmp = dep->m_next;
    delete dep;
    dep = tmp;
  }
  delete m_lock;
}

//_____________________________________________________________________________
//
void
DetailedTasks::assignMessageTags( int me )
{
  // maps from, to (process) pairs to indices for each batch of that pair
  std::map<std::pair<int, int>, int> perPairBatchIndices;

  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    DependencyBatch* batch = m_dep_batches[i];
    int from = batch->m_from_task->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, m_proc_group->size());
    int to = batch->m_to_rank;
    ASSERTRANGE(to, 0, m_proc_group->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      std::pair<int, int> fromToPair = std::make_pair(from, to);
      m_dep_batches[i]->m_message_tag = ++perPairBatchIndices[fromToPair];  // start with one
      if (messagedbg.active()) {
        coutLock.lock();
        messagedbg << "Rank-" << me << " assigning message tag " << batch->m_message_tag << " from task "
                   << batch->m_from_task->getName() << " to task " << batch->m_to_tasks.front()->getName() << ", rank-" << from
                   << " to rank-" << to << "\n";
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
  m_tasks.push_back(task);
}

//_____________________________________________________________________________
//
void
DetailedTasks::makeDWKeyDatabase()
{
  for (int i = 0; i < (int)m_local_tasks.size(); i++) {
    DetailedTask* task = m_local_tasks[i];
    //for reduction task check modifies other task check computes
    const Task::Dependency *comp = task->getTask()->isReductionTask() ? task->getTask()->getModifies() : task->getTask()->getComputes();
    for (; comp != 0; comp = comp->next) {
      const MaterialSubset* matls = comp->matls ? comp->matls : task->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->var->typeDescription()->getType() == TypeDescription::ReductionVariable || comp->var->typeDescription()->getType()
            == TypeDescription::SoleVariable) {
          m_level_key_DB.insert(comp->var, matl, comp->reductionLevel);
        } else {  // if variables saved on varDB
          const PatchSubset* patches = comp->patches ? comp->patches : task->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            m_var_key_DB.insert(comp->var, matl, patch);
            if (dwdbg.active()) {
              dwdbg << "reserve " << comp->var->getName() << " on Patch " << patch->getID() << ", Matl " << matl << "\n";
            }
          }
        }
      }  //end matls
    }  // end comps
  }  // end localtasks
}

//_____________________________________________________________________________
//
void
DetailedTasks::computeLocalTasks( int me )
{
  if (m_local_tasks.size() != 0) {
    return;
  }

  int order = 0;
  m_initially_ready_tasks = TaskQueue();
  for (int i = 0; i < (int)m_tasks.size(); i++) {
    DetailedTask* task = m_tasks[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, m_proc_group->size());
    if (task->getAssignedResourceIndex() == me || task->getTask()->getType() == Task::Reduction) {
      m_local_tasks.push_back(task);

      if (task->areInternalDependenciesSatisfied()) {
        m_initially_ready_tasks.push(task);

        if (mixedDebug.active()) {
          cerrLock.lock();
          mixedDebug << "Initially Ready Task: " << task->getTask()->getName() << "\n";
          cerrLock.unlock();
        }
      }
      task->assignStaticOrder(++order);
    }
  }
}

//_____________________________________________________________________________
//
DetailedTask::DetailedTask(       Task           * task
                          , const PatchSubset    * patches
                          , const MaterialSubset * matls
                          ,       DetailedTasks  * taskGroup
                          )
  : m_task{ task }
  , m_patches{ patches }
  , m_matls{ matls }
  , m_comp_head{ 0 }
  , m_internal_comp_head{ 0 }
  , m_task_group{ taskGroup }
  , m_initiated{ false }
  , m_externally_ready{ false }
  , m_external_dependency_count{ 0 }
  , m_num_pending_internal_dependencies{ 0 }
  , m_internal_dependency_lock{ "DetailedTask Internal Dependencies" }
  , m_resource_index{ -1 }
  , m_static_order{ -1 }
  , m_profile_type{ Normal }
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
void
DetailedTask::doit( const ProcessorGroup                      * pg
                  ,       std::vector<OnDemandDataWarehouseP> & oddws
                  ,       std::vector<DataWarehouseP>         & dws
                  ,       Task::CallBackEvent                   event /* = Task::CPU */
                  )
{
  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "DetailedTask " << this << " begin doit()\n";
    mixedDebug << " task is " << m_task << "\n";
    mixedDebug << "   num Pending Deps: " << m_num_pending_internal_dependencies << "\n";
    mixedDebug << "   Originally needed deps (" << m_internal_dependencies.size() << "):\n";

    std::list<InternalDependency>::iterator iter = m_internal_dependencies.begin();

    for (int i = 0; iter != m_internal_dependencies.end(); iter++, i++) {
      mixedDebug << i << ":    " << *((*iter).m_prerequisite_task->getTask()) << "\n";
    }
    cerrLock.unlock();
  }

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
                 getCUDAStream(currentDevice), currentDevice);
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
void
DetailedTasks::initializeScrubs( std::vector<OnDemandDataWarehouseP> & dws
                               , int                                   dwmap[]
                               )
{
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " Begin initialize scrubs\n";
  }

  std::vector<bool> initialized(dws.size(), false);
  for (int i = 0; i < (int)Task::TotalDWs; i++) {
    if (dwmap[i] < 0) {
      continue;
    }
    OnDemandDataWarehouse* dw = dws[dwmap[i]].get_rep();
    // TODO APH - clean this up (01/31/15)
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
        dw->initializeScrubs(i, &(m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count), initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD="
                 << 1 << '\n';
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &(m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count), true);
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
DetailedTask::scrub( std::vector<OnDemandDataWarehouseP> & dws )
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
      if (scrubmode == DataWarehouse::ScrubComplete || (scrubmode == DataWarehouse::ScrubNonPermanent
          && initialRequires.find(req->var) == initialRequires.end())) {

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
          } else {
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
              } catch (UnknownVariable& e) {
                std::cout << "   BAD BOY FROM Task : " << *this << " scrubbing " << *req << " PATCHES: " << *patches.get_rep()
                          << std::endl;
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
    if (scrubmode == DataWarehouse::ScrubComplete || (scrubmode == DataWarehouse::ScrubNonPermanent
        && initialRequires.find(mod->var) == initialRequires.end())) {

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
      if (scrubmode == DataWarehouse::ScrubComplete || (scrubmode == DataWarehouse::ScrubNonPermanent
          && initialRequires.find(comp->var) == initialRequires.end())) {
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
            } else {
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
// used to be in terms of the dw index within the scheduler,
// but now store WhichDW.  This enables multiple tg execution
void
DetailedTasks::addScrubCount( const VarLabel * var
                            ,       int        matlindex
                            , const Patch    * patch
                            ,       int        dw
                            )
{
  if (patch->isVirtual())
    patch = patch->getRealPatch();
  ScrubItem key(var, matlindex, patch, dw);
  ScrubItem* result;
  result = (m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count).lookup(&key);
  if (!result) {
    result = scinew ScrubItem(var, matlindex, patch, dw);
    (m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count).insert(result);
  }
  result->count++;
  if (scrubout.active() && (var->getName() == dbgScrubVar || dbgScrubVar == "")
      && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1)) {
    scrubout << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/" << patch->getID() << "/" << matlindex
             << "/" << *var << ": " << result->count << std::endl;
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::setScrubCount( const Task::Dependency              * req
                            ,       int                             matl
                            , const Patch                         * patch
                            , std::vector<OnDemandDataWarehouseP> & dws
                            )
{
  ASSERT(!patch->isVirtual());
  DataWarehouse::ScrubMode scrubmode = dws[req->mapDataWarehouse()]->getScrubMode();
  const std::set<const VarLabel*, VarLabel::Compare>& initialRequires = getSchedulerCommon()->getInitialRequiredVars();
  if (scrubmode == DataWarehouse::ScrubComplete || (scrubmode == DataWarehouse::ScrubNonPermanent
      && initialRequires.find(req->var) == initialRequires.end())) {
    int scrubcount;
    if (!getScrubCount(req->var, matl, patch, req->whichdw, scrubcount)) {
      SCI_THROW(InternalError("No scrub count for received MPIVariable: "+req->var->getName(), __FILE__, __LINE__));
    }

    if (scrubout.active() && (req->var->getName() == dbgScrubVar || dbgScrubVar == "")
        && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1)) {
      scrubout << Parallel::getMPIRank() << " setting scrubcount for recv of " << req->mapDataWarehouse() << "/" << patch->getID()
               << "/" << matl << "/" << req->var->getName() << ": " << scrubcount << '\n';
    }
    dws[req->mapDataWarehouse()]->setScrubCount(req->var, matl, patch, scrubcount);
  }
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getScrubCount( const VarLabel * label
                            ,       int        matlIndex
                            , const Patch    * patch
                            ,       int        dw
                            ,       int      & count
                            )
{
  ASSERT(!patch->isVirtual());
  ScrubItem key(label, matlIndex, patch, dw);
  ScrubItem* result = (m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count).lookup(&key);
  if (result) {
    count = result->count;
    return true;
  } else {
    return false;
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::createScrubCounts()
{
  // Clear old ScrubItems
  (m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count).remove_all();

  // Go through each of the tasks and determine which variables it will require
  for (int i = 0; i < (int)m_local_tasks.size(); i++) {
    DetailedTask* dtask = m_local_tasks[i];
    const Task* task = dtask->getTask();
    for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->next) {
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->whichdw;
      TypeDescription::Type type = req->var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          Patch::selectType neighbors;
          IntVector low, high;
          patch->computeVariableExtents(type, req->var->getBoundaryLayer(), req->gtype, req->numGhostCells, neighbors, low, high);
          for (int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];
            for (int m = 0; m < matls->size(); m++) {
              addScrubCount(req->var, matls->get(m), neighbor, whichdw);
            }
          }
        }
      }
    }

    // determine which variables this task will modify
    for (const Task::Dependency* req = task->getModifies(); req != 0; req = req->next) {
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->whichdw;
      TypeDescription::Type type = req->var->typeDescription()->getType();
      if (type != TypeDescription::ReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            addScrubCount(req->var, matls->get(m), patch, whichdw);
          }
        }
      }
    }
  }
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " scrub counts:\n";
    scrubout << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for (FastHashTableIter<ScrubItem> iter(&(m_first_task ? m_first_task->m_scrub_table_count : m_scrub_table_count)); iter.ok();
        ++iter) {
      const ScrubItem* rec = iter.get_key();
      scrubout << rec->dw << '/' << (rec->patch ? rec->patch->getID() : 0) << '/' << rec->matl << '/' << rec->label->getName()
               << "\t\t" << rec->count << '\n';
    }
    scrubout << Parallel::getMPIRank() << " end scrub counts\n";
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::findRequiringTasks( const VarLabel           * var
                                , std::list<DetailedTask*> & requiringTasks
                                )
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
    if (internalDepIter->second->m_variables.find(var) != internalDepIter->second->m_variables.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

//_____________________________________________________________________________
//
DetailedDependency*
DetailedTasks::findMatchingDetailedDep(       DependencyBatch     * batch
                                      ,       DetailedTask        * toTask
                                      ,       Task::Dependency    * req
                                      , const Patch               * fromPatch
                                      ,       int                   matl
                                      ,       IntVector             low
                                      ,       IntVector             high
                                      ,       IntVector           & totalLow
                                      ,       IntVector           & totalHigh
                                      ,       DetailedDependency *& parent_dep
                                      )
{
  totalLow = low;
  totalHigh = high;
  DetailedDependency* dep = batch->m_head;

  parent_dep = 0;
  DetailedDependency* last_dep = 0;
  DetailedDependency* valid_dep = 0;

  //search each dep
  for (; dep != 0; dep = dep->m_next) {
    //if deps are equivalent
    if (fromPatch == dep->m_from_patch && matl == dep->m_matl
        && (req == dep->m_req || (req->var->equals(dep->m_req->var) && req->mapDataWarehouse() == dep->m_req->mapDataWarehouse()))) {

      // total range - the same var in each dep needs to have the same patchlow/high
      dep->m_patch_low = totalLow = Min(totalLow, dep->m_patch_low);
      dep->m_patch_high = totalHigh = Max(totalHigh, dep->m_patch_high);

      int ngcDiff = req->numGhostCells > dep->m_req->numGhostCells ? (req->numGhostCells - dep->m_req->numGhostCells) : 0;
      IntVector new_l = Min(low, dep->m_low);
      IntVector new_h = Max(high, dep->m_high);
      IntVector newRange = new_h - new_l;
      IntVector requiredRange = high - low;
      IntVector oldRange = dep->m_high - dep->m_low + IntVector(ngcDiff, ngcDiff, ngcDiff);
      int newSize = newRange.x() * newRange.y() * newRange.z();
      int requiredSize = requiredRange.x() * requiredRange.y() * requiredRange.z();
      int oldSize = oldRange.x() * oldRange.y() * oldRange.z();

      bool extraComm = newSize > requiredSize + oldSize;

      if (m_sched->useSmallMessages()) {
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
          } else {
            //only take the dep if we haven't already found one
            if (valid_dep == 0) {
              parent_dep = last_dep;
              valid_dep = dep;
            }
            //keep searching in case there is a better dep to combine with
          }
        } else if (dbg.active()) {
          dbg << m_proc_group->myrank() << "            Ignoring: " << dep->m_low << " " << dep->m_high << ", fromPatch = ";
          if (fromPatch) {
            dbg << fromPatch->getID() << '\n';
          } else {
            dbg << "NULL\n";
          }
          dbg << m_proc_group->myrank() << " TP: " << totalLow << " " << totalHigh << std::endl;
        }
      } else {
        if (extraComm) {
          m_extra_communication += newSize - (requiredSize + oldSize);
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
DetailedTasks::possiblyCreateDependency(       DetailedTask                    * from
                                       ,       Task::Dependency                * comp
                                       , const Patch                           * fromPatch
                                       ,       DetailedTask                    * to
                                       ,       Task::Dependency                * req
                                       , const Patch                           * toPatch
                                       ,       int                               matl
                                       , const IntVector                       & low
                                       , const IntVector                       & high
                                       ,       DetailedDependency::CommCondition cond
                                       )
{
  ASSERTRANGE(from->getAssignedResourceIndex(), 0, m_proc_group->size());
  ASSERTRANGE(to->getAssignedResourceIndex(), 0, m_proc_group->size());

  if (dbg.active()) {
    cerrLock.lock();
    {
      dbg << m_proc_group->myrank() << "          " << *to << " depends on " << *from << "\n";
      if (comp) {
        dbg << m_proc_group->myrank() << "            From comp " << *comp;
      } else {
        dbg << m_proc_group->myrank() << "            From OldDW ";
      }
      dbg << " to req " << *req << '\n';
    }
    cerrLock.unlock();
  }

  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();

  // if neither task talks to this processor, return
  if (fromresource != m_proc_group->myrank() && toresource != m_proc_group->myrank()) {
    return;
  }

  if ((toresource == m_proc_group->myrank() || (req->patches_dom != Task::ThisLevel && fromresource == m_proc_group->myrank())) && fromPatch
      && !req->var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->var, matl, fromPatch, req->whichdw);
  }

  //if the dependency is on the same processor then add an internal dependency
  if (fromresource == m_proc_group->myrank() && fromresource == toresource) {
    to->addInternalDependency(from, req->var);

    //In case of multiple GPUs per node, we don't return.  Multiple GPUs
    //need internal dependencies to communicate data.
    if (!Uintah::Parallel::usingDevice()) {
      return;
    }

  }

  //this should have been pruned out earlier
  ASSERT(!req->var->typeDescription()->isReductionVariable())
  // Do not check external deps on SoleVariable
  if (req->var->typeDescription()->getType() == TypeDescription::SoleVariable) {
    return;
  }
#ifdef HAVE_CUDA
  if (Uintah::Parallel::usingDevice()) {
    if (fromresource == m_proc_group->myrank() && fromresource == toresource) {
      if (m_from_patch != toPatch) {
        //printf("In DetailedTasks::createInternalDependencyBatch creating internal dependency from patch %d to patch %d, from task %p to task %p\n", fromPatch->getID(), toPatch->getID(), from, to);
        createInternalDependencyBatch(from, m_comp, m_from_patch, m_to_rank, m_req, toPatch, m_matl, m_low, m_high, cond);
      }
      return;  //We've got all internal dependency information for the GPU, now we can return.
    }
  }
#endif

  //make keys for MPI messages
  if (fromPatch)
    m_var_key_DB.insert(req->var, matl, fromPatch);

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
    m_dep_batches.push_back(batch);
    from->addComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addRequires(batch);
#else
    m_to_rank->addRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active()) {
      dbg << m_proc_group->myrank() << "          NEW BATCH!\n";
    }
  } else if (m_must_consider_internal_dependencies) {  // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add to the batch's toTasks.
      batch->m_to_tasks.push_back(to);
    }
    if (dbg.active()) {
      dbg << m_proc_group->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
    }
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  // create the new dependency
  DetailedDependency* new_dep = scinew DetailedDependency(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  // search for a dependency that can be combined with this dependency

  // location of parent dependency
  DetailedDependency* parent_dep;

  DetailedDependency* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high,
                                                             varRangeLow, varRangeHigh, parent_dep);

  // This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDependency* insert_dep = parent_dep;

  // if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  while (matching_dep != 0) {

    //debugging output
    if (dbg.active()) {
      dbg << m_proc_group->myrank() << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
          << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      dbg << *req->var << '\n';
      dbg << *new_dep->m_req->var << '\n';
      if (comp) {
        dbg << *comp->var << '\n';
      }
      if (new_dep->m_comp) {
        dbg << *new_dep->m_comp->var << '\n';
      }
    }

    // extend the dependency range
    new_dep->m_low = Min(new_dep->m_low, matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);

    // TODO APH - figure this out and clean up (01/31/15)
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
    if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->var->getName() == "p.x") {
        dbg << m_proc_group->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->var
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " "
            << matching_dep->m_high << " cond " << cond << " dw " << req->mapDataWarehouse() << "\n";
      }

      if (fromresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter != m_particle_sends[toresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
//          particleSends_[toresource].erase(pmg);
        }
      } else if (toresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter != m_particle_recvs[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          m_particle_recvs[fromresource].erase(iter);
//          particleRecvs_[fromresource].erase(pmg);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == NULL) {
      batch->m_head = matching_dep->m_next;
    } else {
      parent_dep->m_next = matching_dep->m_next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high, varRangeLow,
                                           varRangeHigh, parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep) {
      insert_dep = parent_dep;
    }
  }

  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->m_patch_low = varRangeLow;
  new_dep->m_patch_high = varRangeHigh;

  if (insert_dep == NULL) {
    //no dependencies are in the list so add it to the head
    batch->m_head = new_dep;
    new_dep->m_next = NULL;
  } else {
    //dependencies already exist so add it at the insert location.
    new_dep->m_next = insert_dep->m_next;
    insert_dep->m_next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
      if (iter == m_particle_sends[toresource].end()) {  //if does not exist
      //add to the sends list
        m_particle_sends[toresource].insert(pmg);
      } else {
        //increment count
        iter->count_++;
      }
    } else if (toresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        //add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      } else {
        //increment the count
        iter->count_++;
      }

    }
    if (req->var->getName() == "p.x") {
      dbg << m_proc_group->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
          << req->mapDataWarehouse() << "\n";
    }
  }

  if (dbg.active()) {
    dbg << m_proc_group->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch) {
      dbg << fromPatch->getID() << '\n';
    } else {
      dbg << "NULL\n";
    }
  }
}


#ifdef HAVE_CUDA

void DetailedTasks::createInternalDependencyBatch(DetailedTask* from,
    Task::Dependency* m_comp,
    const Patch* m_from_patch,
    DetailedTask* m_to_rank,
    Task::Dependency* m_req,
    const Patch *toPatch,
    int m_matl,
    const IntVector& m_low,
    const IntVector& m_high,
    DetailedDependency::CommCondition cond) {
  //get dependancy batch
  DependencyBatch* batch = from->getInternalComputes();
  int toresource = m_to_rank->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();
  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource)
      break;
  }

  //if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, m_to_rank);
    m_dep_batches.push_back(batch);  //Should be fine to push this batch on here, at worst
                                //MPI message tags are created for these which won't get used.
    from->addInternalComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = m_to_rank->addInternalRequires(batch);
#else
    m_to_rank->addInternalRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active())
      dbg << m_proc_group->myrank() << "          NEW BATCH!\n";
  } else if (m_must_consider_internal_dependencies) {  // i.e. threaded mode
    if (m_to_rank->addInternalRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->m_to_tasks.push_back(m_to_rank);
    }
    if (dbg.active())
      dbg << m_proc_group->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  //create the new dependency
  DetailedDependency* new_dep = scinew DetailedDependency(batch->m_head, m_comp, m_req, m_to_rank, m_from_patch, m_matl, m_low, m_high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  DetailedDependency* parent_dep;

  DetailedDependency* matching_dep = findMatchingInternalDetailedDep(batch, m_to_rank, m_req, m_from_patch, m_matl, new_dep->m_low, new_dep->m_high, varRangeLow,
                                                      varRangeHigh, parent_dep);

  //This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDependency* insert_dep = parent_dep;
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
      dbg << m_proc_group->myrank() << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
          << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      dbg << *m_req->var << '\n';
      dbg << *new_dep->m_req->var << '\n';
      if (m_comp)
        dbg << *m_comp->var << '\n';
      if (new_dep->m_comp)
        dbg << *new_dep->m_comp->var << '\n';
    }



    //extend the dependency range
    new_dep->m_low = Min(new_dep->m_low, matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);


    //copy matching dependencies toTasks to the new dependency
    new_dep->m_to_tasks.splice(new_dep->m_to_tasks.begin(), matching_dep->m_to_tasks);

    //erase particle sends/recvs
    if (m_req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && m_req->whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(m_from_patch, m_matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (m_req->var->getName() == "p.x")
        dbg << m_proc_group->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *m_req->var
            << " on patch " << m_from_patch->getID() << " matl " << m_matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
            << " cond " << cond << " dw " << m_req->mapDataWarehouse() << std::endl;

      if (fromresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter!=m_particle_sends[toresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
          //particleSends_[toresource].erase(pmg);
        }
      } else if (toresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter!=m_particle_recvs[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          m_particle_recvs[fromresource].erase(iter);
          //particleRecvs_[fromresource].erase(pmg);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == NULL) {
      batch->m_head = matching_dep->m_next;
    } else {
      parent_dep->m_next = matching_dep->m_next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingInternalDetailedDep(batch, m_to_rank, m_req, m_from_patch, m_matl, new_dep->m_low, new_dep->m_high, varRangeLow, varRangeHigh,
                                           parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep)
      insert_dep = parent_dep;
  }



  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->m_patch_low = varRangeLow;
  new_dep->m_patch_high = varRangeHigh;


  if (insert_dep == NULL) {
    //no dependencies are in the list so add it to the head
    batch->m_head = new_dep;
    new_dep->m_next = NULL;
  } else {
    //depedencies already exist so add it at the insert location.
    new_dep->m_next = insert_dep->m_next;
    insert_dep->m_next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (m_req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && m_req->whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(m_from_patch, m_matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
      if (iter == m_particle_sends[toresource].end())  //if does not exist
          {
        //add to the sends list
        m_particle_sends[toresource].insert(pmg);
      } else {
        //increment count
        iter->count_++;
      }
    } else if (toresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        //add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      } else {
        //increment the count
        iter->count_++;
      }

    }
    if (m_req->var->getName() == "p.x")
      dbg << m_proc_group->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << m_from_patch->getID() << " matl " << m_matl << " range " << m_low << " " << m_high << " cond " << cond << " dw "
          << m_req->mapDataWarehouse() << std::endl;
  }

  if (dbg.active()) {
    dbg << m_proc_group->myrank() << "            ADDED " << m_low << " " << m_high << ", fromPatch = ";
    if (m_from_patch)
      dbg << m_from_patch->getID() << '\n';
    else
      dbg << "NULL\n";
  }

}

DetailedDependency* DetailedTasks::findMatchingInternalDetailedDep(DependencyBatch* batch,
                                                    DetailedTask* toTask,
                                                    Task::Dependency* m_req,
                                                    const Patch* m_from_patch,
                                                    int m_matl,
                                                    IntVector m_low,
                                                    IntVector m_high,
                                                    IntVector& totalLow,
                                                    IntVector& totalHigh,
                                                    DetailedDependency* &parent_dep)
{
  totalLow = m_low;
  totalHigh = m_high;
  DetailedDependency* dep = batch->m_head;

  parent_dep = 0;
  DetailedDependency* last_dep = 0;
  DetailedDependency* valid_dep = 0;
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
  if (m_send_old_data_map.find(proc) == m_send_old_data_map.end()) {
    std::cout << m_proc_group->myrank() << " Error trying to get oldDWSendTask for processor: " << proc
              << " but it does not exist\n";
    throw InternalError("oldDWSendTask does not exist", __FILE__, __LINE__);
  }
#endif 
  return m_tasks[m_send_old_data_map[proc]];
}

//_____________________________________________________________________________
//
void
DetailedTask::addComputes( DependencyBatch* comp )
{
  comp->m_comp_next = m_comp_head;
  m_comp_head = comp;
}

//_____________________________________________________________________________
//
bool
DetailedTask::addRequires( DependencyBatch* req )
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
  return m_internal_reqs.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
// can be called in one of two places - when the last MPI Recv has completed, or from MPIScheduler
void
DetailedTask::checkExternalDepCount()
{
  if (mpidbg.active()) {
    cerrLock.lock();
    mpidbg << "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " external deps: "
           << m_external_dependency_count << " internal deps: " << m_num_pending_internal_dependencies << "\n";
    cerrLock.unlock();
  }

  if (m_external_dependency_count == 0 && m_task_group->m_sched->useInternalDeps() && m_initiated && !m_task->usesMPI()) {
    m_task_group->m_mpi_completed_queue_lock.lock();
    if (mpidbg.active()) {
      cerrLock.lock();
      mpidbg << "Rank-" << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
             << " MPI requirements satisfied, placing into external ready queue\n";
      cerrLock.unlock();
    }

    if (m_externally_ready == false) {
      m_task_group->m_mpi_completed_tasks.push(this);
      m_externally_ready = true;
    }
    m_task_group->m_mpi_completed_queue_lock.unlock();
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::resetDependencyCounts()
{
  m_external_dependency_count = 0;
  m_externally_ready          = false;
  m_initiated                 = false;
}

//_____________________________________________________________________________
//
void
DetailedTask::addInternalDependency(       DetailedTask * prerequisiteTask
                                   , const VarLabel     * var
                                   )
{
  if (m_task_group->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    std::map<DetailedTask*, InternalDependency*>::iterator foundIt = prerequisiteTask->m_internal_dependents.find(this);
    if (foundIt == prerequisiteTask->m_internal_dependents.end()) {
      m_internal_dependencies.push_back(InternalDependency(prerequisiteTask, this, var, 0/* not satisfied */));
      prerequisiteTask->m_internal_dependents[this] = &m_internal_dependencies.back();
      m_num_pending_internal_dependencies = m_internal_dependencies.size();
      if (internaldbg.active()) {
        internaldbg << Parallel::getMPIRank() << " Adding dependency between " << *this << " and " << *prerequisiteTask << "\n";
      }

    } else {
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
cudaStream_t* DetailedTask::getCUDAStream() const
{
  return getCUDAStream(0);
}*/

cudaStream_t* DetailedTask::getCUDAStream(unsigned int deviceNum) const
{
  std::map <unsigned int, cudaStream_t*>::const_iterator it;
  it = d_cudaStreams.find(deviceNum);
  if (it != d_cudaStreams.end()) {
    return it->second;
  }
  return NULL;
}


//void DetailedTask::setCUDAStream(cudaStream_t* s)
//{
//  //d_cudaStream = s;
//  setCUDAStream(0, s);
//};

void DetailedTask::setCUDAStream(unsigned int deviceNum, cudaStream_t* s)
{
  if (s == NULL) {
    d_cudaStreams.erase(deviceNum);
  } else {
    //printf("For device %d inserting stream at %p\n", deviceNum, s);
    //TODO: Needs a write lock.
    d_cudaStreams.insert(std::pair<unsigned int, cudaStream_t*>(deviceNum,s));
  }
};

void DetailedTask::clearCUDAStreams() {
  d_cudaStreams.clear();
}

bool DetailedTask::checkCUDAStreamDone() const
{
  //Check all
  cudaError_t retVal;
  for (std::map<unsigned int, cudaStream_t*>::const_iterator it = d_cudaStreams.begin(); it != d_cudaStreams.end(); ++it) {

    OnDemandDataWarehouse::uintahSetCudaDevice(it->m_first_task);
    retVal = cudaStreamQuery(*(it->second));
    if (retVal == cudaSuccess) {
    //  cout << "checking cuda stream " << d_cudaStream << "ready" << endl;
      continue;
    } else if (retVal == cudaErrorNotReady ) {

      retVal = cudaStreamQuery(*(it->second));
      return false;
    }
    else if (retVal ==  cudaErrorLaunchFailure) {
      printf("ERROR! - DetailedTask::checkCUDAStreamDone() - CUDA kernel execution failure on Task: %s\n", getName().c_str());
      SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task:"+ getName() , __FILE__, __LINE__));
      return false;
    } else { //other error
      printf("ERROR! - DetailedTask::checkCUDAStreamDone() - The stream %p had this error code %d.  This could mean that something else in the stream just hit an error.\n", it->second, retVal);
      SCI_THROW(InternalError("ERROR! - Invalid stream query", __FILE__, __LINE__));
      return false;
    }

  }
  return true;
}

bool DetailedTask::checkCUDAStreamDone(unsigned int deviceNum_) const
{
  // sets the CUDA context, for the call to cudaEventQuery()
  cudaError_t retVal;
  OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum_);
  std::map<unsigned int, cudaStream_t*>::const_iterator it= d_cudaStreams.find(deviceNum_);
  retVal = cudaStreamQuery(*(it->second));
  if (retVal == cudaSuccess) {
//  cout << "checking cuda stream " << d_cudaStream << "ready" << endl;
    return true;
  }
  else if (retVal == cudaErrorNotReady ) {
    return false;
  }
  else if (retVal ==  cudaErrorLaunchFailure) {
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task:"+ getName() , __FILE__, __LINE__));
    return false;
  } else { //other error
    printf("Waiting for 60\n");
    Time::waitFor( (double)60 );
    CUDA_RT_SAFE_CALL (retVal);
    return false;
  }
}

bool DetailedTask::checkAllCUDAStreamsDone() const
{

  // sets the CUDA context, for the call to cudaEventQuery()
  bool retVal = false;

  for (std::map<unsigned int ,cudaStream_t*>::const_iterator it=d_cudaStreams.begin(); it!=d_cudaStreams.end(); ++it){
    retVal = checkCUDAStreamDone(it->m_first_task);
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

#endif // HAVE_CUDA

//_____________________________________________________________________________
//
void
DetailedTask::done( std::vector<OnDemandDataWarehouseP> & dws )
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "This: " << this << " is done with task: " << m_task << "\n";
    mixedDebug << "Name is: " << m_task->getName() << " which has (" << m_internal_dependents.size() << ") tasks waiting on it:\n";
    cerrLock.unlock();
  }

  int cnt = 1000;
  std::map<DetailedTask*, InternalDependency*>::iterator iter;
  for (iter = m_internal_dependents.begin(); iter != m_internal_dependents.end(); iter++) {
    InternalDependency* dep = (*iter).second;

    if (internaldbg.active()) {
      internaldbg << Parallel::getMPIRank() << " Dependency satisfied between " << *dep->m_dependent_task << " and " << *this
                  << "\n";
    }

    dep->m_dependent_task->dependencySatisfied(dep);
    cnt++;
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::dependencySatisfied( InternalDependency * dep )
{
  m_internal_dependency_lock.lock();
  ASSERT(m_num_pending_internal_dependencies > 0);
  unsigned long currentGeneration = m_task_group->getCurrentDependencyGeneration();

  // if false, then the dependency has already been satisfied
  ASSERT(dep->m_satisfied_generation < currentGeneration);

  dep->m_satisfied_generation = currentGeneration;
  m_num_pending_internal_dependencies--;

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << *(dep->m_dependent_task->getTask()) << " has " << m_num_pending_internal_dependencies << " left.\n";
    cerrLock.unlock();
  }

  if (internaldbg.active()) {
    internaldbg << Parallel::getMPIRank() << " satisfying dependency: prereq: " << *dep->m_prerequisite_task << " dep: "
                << *dep->m_dependent_task << " numPending: " << m_num_pending_internal_dependencies << "\n";
  }

  if (m_num_pending_internal_dependencies == 0) {
    m_task_group->internalDependenciesSatisfied(this);
    // reset for next timestep
    m_num_pending_internal_dependencies = m_internal_dependencies.size();
  }
  m_internal_dependency_lock.unlock();
}

//_____________________________________________________________________________
//
namespace Uintah {

std::ostream&
operator<<(       std::ostream & out
          , const DetailedTask & task
          )
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
      } else {
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
    } else {
      out << task.getAssignedResourceIndex();
    }
//#ifdef HAVE_CUDA
//    if( task.getCUDAStream() ){
//      out << std::hex << " using CUDA stream " << task.getCUDAStream() << std::dec;
//    }
//#endif

  }
  coutLock.unlock();

  return out;
}

//_____________________________________________________________________________
//
std::ostream&
operator<<(       std::ostream       & out
          , const DetailedDependency & dep
          )
{
  coutLock.lock();
  {
    out << dep.m_req->var->getName();
    if (dep.isNonDataDependency()) {
      out << " non-data dependency";
    } else {
      out << " on patch " << dep.m_from_patch->getID();
    }
    out << ", matl " << dep.m_matl << ", low=" << dep.m_low << ", high=" << dep.m_high;
  }
  coutLock.unlock();

  return out;
}

} // namespace Uintah

//_____________________________________________________________________________
//
void
DetailedTasks::internalDependenciesSatisfied( DetailedTask * task )
{
  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "Begin internalDependenciesSatisfied\n";
    cerrLock.unlock();
  }
  m_ready_queue_lock.lock();
  {
    m_ready_tasks.push(task);

    if (mixedDebug.active()) {
      cerrLock.lock();
      mixedDebug << *task << " satisfied.  Now " << m_ready_tasks.size() << " ready.\n";
      cerrLock.unlock();
    }
  }
  m_ready_queue_lock.unlock();
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  m_ready_queue_lock.lock();
  {
    if (!m_ready_tasks.empty()) {
      nextTask = m_ready_tasks.front();
      m_ready_tasks.pop();
    }
  }
  m_ready_queue_lock.unlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numInternalReadyTasks()
{
  int size = 0;
  m_ready_queue_lock.lock();
  {
    size = m_ready_tasks.size();
  }
  m_ready_queue_lock.unlock();
  return size;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  m_mpi_completed_queue_lock.lock();
  {
    if (!m_mpi_completed_tasks.empty()) {
      nextTask = m_mpi_completed_tasks.top();
      m_mpi_completed_tasks.pop();
    }
  }
  m_mpi_completed_queue_lock.unlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numExternalReadyTasks()
{
  int size = 0;
  m_mpi_completed_queue_lock.lock();
  {
    size = m_mpi_completed_tasks.size();
  }
  m_mpi_completed_queue_lock.unlock();
  return size;
}


#ifdef HAVE_CUDA
//_____________________________________________________________________________
//
DetailedTask* DetailedTasks::getNextFinalizeDevicePreparationTask()
{
  DetailedTask* nextTask = NULL;
  deviceFinalizePreparationQueueLock_.writeLock();
  if (!finalizeDevicePreparationTasks_.empty()) {
    nextTask = finalizeDevicePreparationTasks_.top();
    finalizeDevicePreparationTasks_.pop();
  }
  deviceFinalizePreparationQueueLock_.writeUnlock();
  return nextTask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInitiallyReadyDeviceTask()
{
  DetailedTask* nextTask = NULL;
  deviceReadyQueueLock_.writeLock();
  {
    if (!initiallyReadyDeviceTasks_.empty()) {
      nextTask = initiallyReadyDeviceTasks_.top();
      initiallyReadyDeviceTasks_.pop();
    }
  }
  deviceReadyQueueLock_.writeUnlock();

  return nextTask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextCompletionPendingDeviceTask()
{
  DetailedTask* nextTask = NULL;
  deviceCompletedQueueLock_.writeLock();
  if (!completionPendingDeviceTasks_.empty()) {
    nextTask = completionPendingDeviceTasks_.top();
    completionPendingDeviceTasks_.pop();
  }
  deviceCompletedQueueLock_.writeUnlock();

  return nextTask;
}

//_____________________________________________________________________________
//
DetailedTask* DetailedTasks::getNextInitiallyReadyHostTask()
{
  DetailedTask* nextTask = NULL;
  hostReadyQueueLock_.writeLock();
  if (!initiallyReadyHostTasks_.empty()) {
    nextTask = initiallyReadyHostTasks_.top();
    initiallyReadyHostTasks_.pop();
  }
  hostReadyQueueLock_.writeUnlock();

  return nextTask;
}

//_____________________________________________________________________________
//
DetailedTask* DetailedTasks::peekNextFinalizeDevicePreparationTask()
{
  deviceFinalizePreparationQueueLock_.readLock();
  DetailedTask* dtask = finalizeDevicePreparationTasks_.top();
  deviceFinalizePreparationQueueLock_.readUnlock();

  return dtask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::peekNextInitiallyReadyDeviceTask()
{
  deviceReadyQueueLock_.readLock();
  DetailedTask* dtask = initiallyReadyDeviceTasks_.top();
  deviceReadyQueueLock_.readUnlock();

  return dtask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::peekNextCompletionPendingDeviceTask()
{
  DetailedTask* dtask = NULL;
  deviceCompletedQueueLock_.readLock();
  {
    dtask = completionPendingDeviceTasks_.top();
  }
  deviceCompletedQueueLock_.readUnlock();

  return dtask;
}

//_____________________________________________________________________________
//
DetailedTask* DetailedTasks::peekNextInitiallyReadyHostTask()
{
  hostReadyQueueLock_.readLock();
  DetailedTask* dtask = initiallyReadyHostTasks_.top();
  hostReadyQueueLock_.readUnlock();

  return dtask;
}

//_____________________________________________________________________________
//
void DetailedTasks::addFinalizeDevicePreparation(DetailedTask* dtask)
{
  deviceFinalizePreparationQueueLock_.writeLock();
  finalizeDevicePreparationTasks_.push(dtask);
  deviceFinalizePreparationQueueLock_.writeUnlock();
}

//_____________________________________________________________________________
//
void
DetailedTasks::addInitiallyReadyDeviceTask( DetailedTask* dtask )
{
  deviceReadyQueueLock_.writeLock();
  {
    initiallyReadyDeviceTasks_.push(dtask);
  }
  deviceReadyQueueLock_.writeUnlock();
}

//_____________________________________________________________________________
//
void
DetailedTasks::addCompletionPendingDeviceTask( DetailedTask* dtask )
{
  deviceCompletedQueueLock_.writeLock();
  {
    completionPendingDeviceTasks_.push(dtask);
  }
  deviceCompletedQueueLock_.writeUnlock();
}

void DetailedTasks::addInitiallyReadyHostTask(DetailedTask* dtask)
{
  hostReadyQueueLock_.writeLock();
  initiallyReadyHostTasks_.push(dtask);
  hostReadyQueueLock_.writeUnlock();
}


#endif

//_____________________________________________________________________________
//
void
DetailedTasks::initTimestep()
{
  m_ready_tasks = m_initially_ready_tasks;
  incrementDependencyGeneration();
  initializeBatches();
}

//_____________________________________________________________________________
//
void
DetailedTasks::incrementDependencyGeneration()
{
  if (m_current_dependency_generation >= ULONG_MAX) {
    SCI_THROW(InternalError("DetailedTasks::currentDependencySatisfyingGeneration has overflowed", __FILE__, __LINE__));
  }
  m_current_dependency_generation++;
}

//_____________________________________________________________________________
//
void
DetailedTasks::initializeBatches()
{
  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    m_dep_batches[i]->reset();
  }
}

//_____________________________________________________________________________
//
void
DependencyBatch::reset()
{
  if (m_to_tasks.size() > 1) {
    if (m_lock == 0) {
      m_lock = scinew Mutex("DependencyBatch receive lock");
    }
  }
  m_received = false;
  m_made_mpi_request = false;
}

//_____________________________________________________________________________
//
bool
DependencyBatch::makeMPIRequest()
{
  if (m_to_tasks.size() > 1) {
    ASSERT(m_lock != 0);
    if (!m_made_mpi_request) {
      m_lock->lock();
      if (!m_made_mpi_request) {
        m_made_mpi_request = true;
        m_lock->unlock();
        return true;  // first to make the request
      } else {
        m_lock->unlock();
        return false;  // got beat out -- request already made
      }
    }
    return false;  // request already made
  } else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(!m_made_mpi_request);
    m_made_mpi_request = true;
    return true;
  }
}

//_____________________________________________________________________________
//
void
DependencyBatch::received( const ProcessorGroup * pg )
{
  m_received = true;

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "Received batch message " << m_message_tag << " from task " << *m_from_task << "\n";

    for (DetailedDependency* dep = m_head; dep != 0; dep = dep->m_next) {
      mixedDebug << "\tSatisfying " << *dep << "\n";
    }
    cerrLock.unlock();
  }
  if (waitout.active()) {
    //add the time waiting on MPI to the wait times per from task
    s_wait_times[m_from_task->getTask()->getName()] += CommRecMPI::WaitTimePerMessage;
  }
  //set all the toVars to valid, meaning the mpi has been completed
  for (std::vector<Variable*>::iterator iter = m_to_variables.begin(); iter != m_to_variables.end(); iter++) {
    (*iter)->setValid();
  }
  for (std::list<DetailedTask*>::iterator iter = m_to_tasks.begin(); iter != m_to_tasks.end(); iter++) {
    // if the count is 0, the task will add itself to the external ready queue
    //cout << pg->myrank() << "  Dec: " << *fromTask << " for " << *(*iter) << endl;
    (*iter)->decrementExternalDepCount();
    //cout << Parallel::getMPIRank() << "   task " << **(iter) << " received a message, remaining count " << (*iter)->getExternalDepCount() << endl;
    (*iter)->checkExternalDepCount();
  }

  //clear the variables that have outstanding MPI as they are completed now.
  m_to_variables.clear();

  // TODO - FIXME: Figure this out and clean up - APH (02/12/16)
#if 0
  if (!receiveListeners_.empty()) {
    // only needed when multiple tasks need a batch
    ASSERT(m_to_tasks.size() > 1);
    ASSERT(m_lock != 0);
    m_lock->lock();
    {
      for (set<int>::iterator iter = receiveListeners_.begin(); iter != receiveListeners_.end(); ++iter) {
        // send WakeUp messages to threads on the same processor
        MPI_Send(0, 0, MPI_INT, pg->myrank(), *iter, pg->getComm());
      }
      receiveListeners_.clear();
    }
    m_lock->unlock();
  }
#endif
}

//_____________________________________________________________________________
//
void
DetailedTasks::logMemoryUse(       std::ostream  & out
                           ,       unsigned long & total
                           , const std::string   & tag
                           )
{
  std::ostringstream elems1;
  elems1 << m_tasks.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1, elems1.str(), m_tasks.size() * sizeof(DetailedTask), 0);
  std::ostringstream elems2;
  elems2 << m_dep_batches.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1, elems2.str(), m_dep_batches.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    for (DetailedDependency* p = m_dep_batches[i]->m_head; p != 0; p = p->m_next) {
      ndeps++;
    }
  }
  std::ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "DetailedDep", 0, -1, elems3.str(), ndeps * sizeof(DetailedDependency), 0);
}

//_____________________________________________________________________________
//
void
DetailedTasks::emitEdges( ProblemSpecP edgesElement
                        ,  int         rank
                        )
{
  for (int i = 0; i < (int)m_tasks.size(); i++) {
    ASSERTRANGE(m_tasks[i]->getAssignedResourceIndex(), 0, m_proc_group->size());
    if (m_tasks[i]->getAssignedResourceIndex() == rank) {
      m_tasks[i]->emitEdges(edgesElement);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::emitEdges( ProblemSpecP edgesElement )
{
  std::map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = m_requires.begin(); req_iter != m_requires.end(); req_iter++) {
    DetailedTask* fromTask = (*req_iter).first->m_from_task;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  std::list<InternalDependency>::iterator iter;
  for (iter = m_internal_dependencies.begin(); iter != m_internal_dependencies.end(); iter++) {
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


class PatchIDIterator {

  public:

    PatchIDIterator(const std::vector<const Patch*>::const_iterator& iter)
      : m_iter{ iter }
    {}

    PatchIDIterator& operator=(const PatchIDIterator& iter2)
    {
      m_iter = iter2.m_iter;
      return *this;
    }

    int operator*()
    {
      const Patch* patch = *m_iter;  //vector<Patch*>::iterator::operator*();
      return patch ? patch->getID() : -1;
    }

    PatchIDIterator& operator++()
    {
      m_iter++;
      return *this;
    }

    bool operator!=(const PatchIDIterator& iter2)
    {
      return m_iter != iter2.m_iter;
    }

  private:
    std::vector<const Patch*>::const_iterator m_iter;
};

//_____________________________________________________________________________
//
std::string
DetailedTask::getName() const
{
  if (m_name != "") {
    return m_name;
  }

  m_name = std::string(m_task->getName());

  if (m_patches != 0) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder(PatchIDIterator(m_patches->getVector().begin()), PatchIDIterator(m_patches->getVector().end()));
    m_name += std::string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if (m_matls != 0) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder(m_matls->getVector().begin(), m_matls->getVector().end());
    m_name += std::string(" (Matls: ") + matlSet.toString() + ")";
  }

  return m_name;
}

//_____________________________________________________________________________
//
// comparing the priority of two detailed tasks True means give rtask priority
bool
DetailedTaskPriorityComparison::operator()( DetailedTask *& ltask
                                          , DetailedTask *& rtask
                                          )
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
    return ltask->getTask()->childTasks.size() < rtask->getTask()->childTasks.size();
  } else if (alg == LeastChildren) {
    return ltask->getTask()->childTasks.size() > rtask->getTask()->childTasks.size();
  } else if (alg == MostAllChildren) {
    return ltask->getTask()->allChildTasks.size() < rtask->getTask()->allChildTasks.size();
  } else if (alg == LeastAllChildren) {
    return ltask->getTask()->allChildTasks.size() > rtask->getTask()->allChildTasks.size();
  } else if (alg == MostL2Children || alg == LeastL2Children) {
    int ll2 = 0;
    int rl2 = 0;
    std::set<Task*>::iterator it;
    for (it = ltask->getTask()->childTasks.begin(); it != ltask->getTask()->childTasks.end(); it++) {
      ll2 += (*it)->childTasks.size();
    }
    for (it = rtask->getTask()->childTasks.begin(); it != rtask->getTask()->childTasks.end(); it++) {
      rl2 += (*it)->childTasks.size();
    }
    if (alg == MostL2Children) {
      return ll2 < rl2;
    } else {
      return ll2 > rl2;
    }
  } else if (alg == MostMessages || alg == LeastMessages) {
    int lmsg = 0;
    int rmsg = 0;
    for (DependencyBatch* batch = ltask->getComputes(); batch != 0; batch = batch->m_comp_next) {
      for (DetailedDependency* dep = batch->m_head; dep != 0; dep = dep->m_next) {
        lmsg++;
      }
    }
    for (DependencyBatch* batch = rtask->getComputes(); batch != 0; batch = batch->m_comp_next) {
      for (DetailedDependency* dep = batch->m_head; dep != 0; dep = dep->m_next) {
        rmsg++;
      }
    }
    if (alg == MostMessages) {
      return lmsg < rmsg;
    } else {
      return lmsg > rmsg;
    }
  } else if (alg == PatchOrder) {  //smaller level, larger size, smaller patchied, smaller tasksortid
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    if (getLevel(lpatches) == getLevel(rpatches)) {
      if (lpatches->size() == rpatches->size()) {
        return lpatches->get(0)->getID() > rpatches->get(0)->getID();
      } else {
        return lpatches->size() < rpatches->size();
      }
    } else {
      return getLevel(lpatches) > getLevel(rpatches);
    }
  } else if (alg == PatchOrderRandom) {  //smaller level, larger size, smaller patchied, smaller tasksortid
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    if (getLevel(lpatches) == getLevel(rpatches)) {
      if (lpatches->size() == rpatches->size()) {
        return (random() % 2 == 0);
      } else {
        return lpatches->size() < rpatches->size();
      }
    } else {
      return getLevel(lpatches) > getLevel(rpatches);
    }
  } else {
    return false;
  }
}
