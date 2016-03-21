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

#include <CCA/Components/Schedulers/DetailedTasks_Exp.cpp>

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

// sync cout/cerr so they are readable when output by multiple threads
#include <sci_defs/config_defs.h>
#include <sci_defs/cuda_defs.h>

extern Uintah::Mutex cerrLock;
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
DetailedTasks::DetailedTasks(       SchedulerCommon* sc,
                              const ProcessorGroup*  pg,
                                    DetailedTasks*   first,
                              const TaskGraph*       taskgraph,
                              const std::set<int>&        neighborhood_processors,
                                    bool             mustConsiderInternalDependencies /* = false */ ) :
  m_scheduler(sc),
  m_proc_group(pg),
  m_first(first),
  m_task_graph(taskgraph),
  m_must_consider_internal_deps(mustConsiderInternalDependencies),
  m_current_dependency_generation(1),
  m_extra_communication(0)
#ifdef HAVE_CUDA
  ,
  deviceVerifyDataTransferCompletionQueueLock_("DetailedTasks Device Verify Data Transfer Queue"),
  deviceFinalizePreparationQueueLock_("DetailedTasks Device Finalize Preparation Queue"),
  deviceReadyQueueLock_("DetailedTasks Device Ready Queue"),
  deviceCompletedQueueLock_("DetailedTasks Device Completed Queue"),
  hostFinalizePreparationQueueLock_("DetailedTasks Host Finalize Preparation Queue"),
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

  m_send_old_data_task = scinew Task( "send old data", Task::InitialSend );
  m_send_old_data_task->d_phase = 0;
  m_send_old_data_task->setMapping( dwmap );

  // Create a send old detailed task for every processor in my neighborhood.
  for (std::set<int>::iterator iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); iter++) {
    DetailedTask* newtask = scinew DetailedTask( m_send_old_data_task, 0, 0, this );
    newtask->assignResource(*iter);
    //use a map because the processors in this map are likely to be sparse
    m_send_old_map[*iter] = m_tasks.size();
    m_tasks.push_back(newtask);
  }
}

//_____________________________________________________________________________
//
DetailedTasks::~DetailedTasks()
{
  // Free dynamically allocated SrubItems
  (m_first ? m_first->m_scrub_count_table : m_scrub_count_table).remove_all();

  for (int i = 0; i < (int)m_dependency_batches.size(); i++) {
    delete m_dependency_batches[i];
  }

  for (int i = 0; i < (int)m_tasks.size(); i++) {
    delete m_tasks[i];
  }
  delete m_send_old_data_task;
}

//_____________________________________________________________________________
//
void
DetailedTasks::assignMessageTags( int me )
{
  // maps from, to (process) pairs to indices for each batch of that pair
  std::map<std::pair<int, int>, int> perPairBatchIndices;

  for (int i = 0; i < (int)m_dependency_batches.size(); i++) {
    DependencyBatch* batch = m_dependency_batches[i];
    int from = batch->m_from_task->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, m_proc_group->size());
    int to = batch->m_to_proc;
    ASSERTRANGE(to, 0, m_proc_group->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      std::pair<int, int> fromToPair = std::make_pair(from, to);
      m_dependency_batches[i]->m_message_tag = ++perPairBatchIndices[fromToPair];  // start with one
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
  m_tasks.push_back(task);
}

//_____________________________________________________________________________
//
void
DetailedTasks::makeDWKeyDatabase()
{
  for (int i = 0; i < (int)m_local_tasks.size(); i++) {
    DetailedTask* dtask = m_local_tasks[i];
    const Task* task = dtask->getTask();
    //for reduction task check modifies other task check computes
    const Task::Dependency *comp = task->isReductionTask() ? task->getModifies() : task->getComputes();
    for (; comp != 0; comp = comp->next) {
      const MaterialSubset* matls = comp->matls ? comp->matls : dtask->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->var->typeDescription()->getType() == TypeDescription::ReductionVariable ||
            comp->var->typeDescription()->getType() == TypeDescription::SoleVariable) {
          m_level_key_DB.insert(comp->var, matl, comp->reductionLevel);
        }
        else { // if variables saved on varDB
          const PatchSubset* patches = comp->patches ? comp->patches : dtask->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            m_var_key_DB.insert(comp->var, matl, patch);
            if (dwdbg.active()) {
              dwdbg << "reserve " << comp->var->getName() << " on Patch " << patch->getID() << ", Matl " << matl << "\n";
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

        if (dbg.active()) {
          cerrLock.lock();
          dbg << "Initially Ready Task: " << task->getTask()->getName() << "\n";
          cerrLock.unlock();
        }
      }
      task->assignStaticOrder(++order);
    }
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::initializeScrubs( std::vector<OnDemandDataWarehouseP>& dws,
                                 int                             dwmap[] )
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
        dw->initializeScrubs(i, &(m_first ? m_first->m_scrub_count_table : m_scrub_count_table), initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD="
                 << 1 << '\n';
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &(m_first ? m_first->m_scrub_count_table : m_scrub_count_table), true);
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
  result = (m_first ? m_first->m_scrub_count_table : m_scrub_count_table).lookup(&key);
  if (!result) {
    result = scinew ScrubItem(var, matlindex, patch, dw);
    (m_first ? m_first->m_scrub_count_table : m_scrub_count_table).insert(result);
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
DetailedTasks::setScrubCount( const Task::Dependency*                    req,
                                    int                                  matl,
                              const Patch*                               patch,
                              std::vector<OnDemandDataWarehouseP>&      dws )
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
DetailedTasks::getScrubCount( const VarLabel* label,
                                    int       matlIndex,
                              const Patch*    patch,
                                    int       dw,
                                    int&      count )
{
  ASSERT(!patch->isVirtual());
  ScrubItem key(label, matlIndex, patch, dw);
  ScrubItem* result = (m_first ? m_first->m_scrub_count_table : m_scrub_count_table).lookup(&key);
  if (result) {
    count = result->count;
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
  (m_first ? m_first->m_scrub_count_table : m_scrub_count_table).remove_all();

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
    for (FastHashTableIter<ScrubItem> iter(&(m_first ? m_first->m_scrub_count_table : m_scrub_count_table)); iter.ok(); ++iter) {
      const ScrubItem* rec = iter.get_key();
      scrubout << rec->dw << '/' << (rec->patch ? rec->patch->getID() : 0) << '/' << rec->matl << '/' << rec->label->getName()
               << "\t\t" << rec->count << '\n';
    }
    scrubout << Parallel::getMPIRank() << " end scrub counts\n";
  }
}

//_____________________________________________________________________________
//

//_____________________________________________________________________________
//
DetailedDependency*
DetailedTasks::findMatchingDetailedDep(       DependencyBatch*  batch,
                                              DetailedTask*     toTask,
                                              Task::Dependency* req,
                                        const Patch*            fromPatch,
                                              int               matl,
                                              IntVector         low,
                                              IntVector         high,
                                              IntVector&        totalLow,
                                              IntVector&        totalHigh,
                                              DetailedDependency*&     parent_dep )
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

      if (m_scheduler->useSmallMessages()) {
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
          dbg << m_proc_group->myrank() << "            Ignoring: " << dep->m_low << " " << dep->m_high << ", fromPatch = ";
          if (fromPatch) {
            dbg << fromPatch->getID() << '\n';
          }
          else {
            dbg << "NULL\n";
          }
          dbg << m_proc_group->myrank() << " TP: " << totalLow << " " << totalHigh << std::endl;
        }
      }
      else {
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
DetailedTasks::possiblyCreateDependency(       DetailedTask*              from,
                                               Task::Dependency*          comp,
                                         const Patch*                     fromPatch,
                                               DetailedTask*              to,
                                               Task::Dependency*          req,
                                         const Patch*                     toPatch,
                                               int                        matl,
                                         const IntVector&                 low,
                                         const IntVector&                 high,
                                               DetailedDependency::CommCondition cond )
{
  ASSERTRANGE(from->getAssignedResourceIndex(), 0, m_proc_group->size());
  ASSERTRANGE(to->getAssignedResourceIndex(),   0, m_proc_group->size());

  if (dbg.active()) {
    cerrLock.lock();
    {
      dbg << m_proc_group->myrank() << "          " << *to << " depends on " << *from << "\n";
      if (comp) {
        dbg << m_proc_group->myrank() << "            From comp " << *comp;
      }
      else {
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

  if ((toresource == m_proc_group->myrank() || (req->patches_dom != Task::ThisLevel && fromresource == m_proc_group->myrank()))
      && fromPatch && !req->var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->var, matl, fromPatch, req->whichdw);
  }

  //if the dependency is on the same processor then add an internal dependency
  if (fromresource == m_proc_group->myrank() && fromresource == toresource) {
    to->addInternalDependency(from, req->var);

    //In case of multiple GPUs per node, we don't return.  Multiple GPUs
    //need internal dependencies to communicate data.
    if ( ! Uintah::Parallel::usingDevice()) {
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
        createInternalDependencyBatch(from, m_comp, m_from_patch, m_to_proc, m_req, toPatch, m_matl, m_low, m_high, cond);
      }
      return; //We've got all internal dependency information for the GPU, now we can return.
    }
  }
#endif

  //make keys for MPI messages
  if (fromPatch) m_var_key_DB.insert(req->var,matl,fromPatch);

  //get dependency batch
  DependencyBatch* batch = from->getComputes();

  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->m_comp_next) {
    if (batch->m_to_proc == toresource) {
      break;
    }
  }

  //if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, to);
    m_dependency_batches.push_back(batch);
    from->addComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addRequires(batch);
#else
    m_to_proc->addRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active()) {
      dbg << m_proc_group->myrank() << "          NEW BATCH!\n";
    }
  }
  else if (m_must_consider_internal_deps) {  // i.e. threaded mode
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

  DetailedDependency* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high, varRangeLow,
                                                      varRangeHigh, parent_dep);

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
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
            << " cond " << cond << " dw " << req->mapDataWarehouse() << "\n";
      }

      if (fromresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter != m_particle_sends[toresource].end());
        //subtract one from the count
        iter->m_count--;
        //if the count is zero erase it from the sends list
        if (iter->m_count == 0) {
          m_particle_sends[toresource].erase(iter);
//          particleSends_[toresource].erase(pmg);
        }
      }
      else if (toresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter != m_particle_recvs[fromresource].end());
        //subtract one from the count
        iter->m_count--;
        //if the count is zero erase it from the recvs list
        if (iter->m_count == 0) {
          m_particle_recvs[fromresource].erase(iter);
//          particleRecvs_[fromresource].erase(pmg);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == NULL) {
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

  if (insert_dep == NULL) {
    //no dependencies are in the list so add it to the head
    batch->m_head = new_dep;
    new_dep->m_next = NULL;
  }
  else {
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
      }
      else {
        //increment count
        iter->m_count++;
      }
    }
    else if (toresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        //add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      }
      else {
        //increment the count
        iter->m_count++;
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
    }
    else {
      dbg << "NULL\n";
    }
  }
}
#ifdef HAVE_CUDA

void DetailedTasks::createInternalDependencyBatch(DetailedTask* from,
    Task::Dependency* m_comp,
    const Patch* m_from_patch,
    DetailedTask* m_to_proc,
    Task::Dependency* m_req,
    const Patch *toPatch,
    int m_matl,
    const IntVector& m_low,
    const IntVector& m_high,
    DetailedDependency::CommCondition cond) {
  //get dependancy batch
  DependencyBatch* batch = from->getInternalComputes();
  int toresource = m_to_proc->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();
  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->m_comp_next) {
    if (batch->m_to_proc == toresource)
      break;
  }

  //if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, m_to_proc);
    m_dependency_batches.push_back(batch);  //Should be fine to push this batch on here, at worst
                                //MPI message tags are created for these which won't get used.
    from->addInternalComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = m_to_proc->addInternalRequires(batch);
#else
    m_to_proc->addInternalRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (dbg.active())
      dbg << m_proc_group->myrank() << "          NEW BATCH!\n";
  } else if (m_must_consider_internal_deps) {  // i.e. threaded mode
    if (m_to_proc->addInternalRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->m_to_tasks.push_back(m_to_proc);
    }
    if (dbg.active())
      dbg << m_proc_group->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  //create the new dependency
  DetailedDependency* new_dep = scinew DetailedDependency(batch->m_head, m_comp, m_req, m_to_proc, m_from_patch, m_matl, m_low, m_high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  DetailedDependency* parent_dep;

  DetailedDependency* matching_dep = findMatchingInternalDetailedDep(batch, m_to_proc, m_req, m_from_patch, m_matl, new_dep->m_low, new_dep->m_high, varRangeLow,
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
        iter->m_count--;
        //if the count is zero erase it from the sends list
        if (iter->m_count == 0) {
          m_particle_sends[toresource].erase(iter);
          //particleSends_[toresource].erase(pmg);
        }
      } else if (toresource == m_proc_group->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter!=m_particle_recvs[fromresource].end());
        //subtract one from the count
        iter->m_count--;
        //if the count is zero erase it from the recvs list
        if (iter->m_count == 0) {
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
    matching_dep = findMatchingInternalDetailedDep(batch, m_to_proc, m_req, m_from_patch, m_matl, new_dep->m_low, new_dep->m_high, varRangeLow, varRangeHigh,
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
        iter->m_count++;
      }
    } else if (toresource == m_proc_group->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        //add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      } else {
        //increment the count
        iter->m_count++;
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
  if (m_send_old_map.find(proc) == m_send_old_map.end()) {
    std::cout << m_proc_group->myrank() << " Error trying to get oldDWSendTask for processor: " << proc << " but it does not exist\n";
    throw InternalError("oldDWSendTask does not exist", __FILE__, __LINE__);
  }
#endif
  return m_tasks[m_send_old_map[proc]];
}





//_____________________________________________________________________________
//
void
DetailedTasks::internalDependenciesSatisfied( DetailedTask* task )
{
  m_ready_queue_lock.lock();
  {
    m_ready_tasks.push(task);
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
DetailedTask*
DetailedTasks::getNextVerifyDataTransferCompletionTask()
{
  DetailedTask* nextTask = NULL;
  deviceVerifyDataTransferCompletionQueueLock_.writeLock();
  {
    if (!verifyDataTransferCompletionTasks_.empty()) {
      nextTask = verifyDataTransferCompletionTasks_.top();
      verifyDataTransferCompletionTasks_.pop();
    }
  }
  deviceVerifyDataTransferCompletionQueueLock_.writeUnlock();

  return nextTask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextFinalizeDevicePreparationTask()
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
DetailedTask*
DetailedTasks::getNextFinalizeHostPreparationTask()
{
  DetailedTask* nextTask = NULL;
  hostFinalizePreparationQueueLock_.writeLock();
  if (!finalizeHostPreparationTasks_.empty()) {
    nextTask = finalizeHostPreparationTasks_.top();
    finalizeHostPreparationTasks_.pop();
  }
  hostFinalizePreparationQueueLock_.writeUnlock();
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
DetailedTask*
DetailedTasks::peekNextVerifyDataTransferCompletionTask()
{
  deviceVerifyDataTransferCompletionQueueLock_.readLock();
  DetailedTask* dtask = verifyDataTransferCompletionTasks_.top();
  deviceVerifyDataTransferCompletionQueueLock_.readUnlock();

  return dtask;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::peekNextFinalizeDevicePreparationTask()
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
DetailedTask*
DetailedTasks::peekNextFinalizeHostPreparationTask()
{
  hostFinalizePreparationQueueLock_.readLock();
  DetailedTask* dtask = finalizeHostPreparationTasks_.top();
  hostFinalizePreparationQueueLock_.readUnlock();

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
void DetailedTasks::addVerifyDataTransferCompletion(DetailedTask* dtask)
{
  deviceVerifyDataTransferCompletionQueueLock_.writeLock();
  verifyDataTransferCompletionTasks_.push(dtask);
  deviceVerifyDataTransferCompletionQueueLock_.writeUnlock();
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

//_____________________________________________________________________________
//
void DetailedTasks::addFinalizeHostPreparation(DetailedTask* dtask)
{
  hostFinalizePreparationQueueLock_.writeLock();
  finalizeHostPreparationTasks_.push(dtask);
  hostFinalizePreparationQueueLock_.writeUnlock();
}

//_____________________________________________________________________________
//
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
  for (int i = 0; i < (int)m_dependency_batches.size(); i++) {
    m_dependency_batches[i]->reset();
  }
}



//_____________________________________________________________________________
//
void
DetailedTasks::logMemoryUse(       std::ostream&       out,
                                   unsigned long& total,
                             const std::string&        tag )
{
  std::ostringstream elems1;
  elems1 << m_tasks.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1, elems1.str(), m_tasks.size() * sizeof(DetailedTask), 0);
  std::ostringstream elems2;
  elems2 << m_dependency_batches.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1, elems2.str(), m_dependency_batches.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)m_dependency_batches.size(); i++) {
    for (DetailedDependency* p = m_dependency_batches[i]->m_head; p != 0; p = p->m_next) {
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
DetailedTasks::emitEdges( ProblemSpecP edgesElement,
                          int          rank )
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
    return ltask->getTask()->childTasks.size() < rtask->getTask()->childTasks.size();
  }
  else if (alg == LeastChildren) {
    return ltask->getTask()->childTasks.size() > rtask->getTask()->childTasks.size();
  }
  else if (alg == MostAllChildren) {
    return ltask->getTask()->allChildTasks.size() < rtask->getTask()->allChildTasks.size();
  }
  else if (alg == LeastAllChildren) {
    return ltask->getTask()->allChildTasks.size() > rtask->getTask()->allChildTasks.size();
  }
  else if (alg == MostL2Children || alg == LeastL2Children) {
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
    }
    else {
      return ll2 > rl2;
    }
  }
  else if (alg == MostMessages || alg == LeastMessages) {
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

} // namespace Uintah

#endif // UINTAH_USING_EXPERIMENTAL
