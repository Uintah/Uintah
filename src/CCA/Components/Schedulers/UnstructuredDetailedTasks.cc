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

#include <CCA/Components/Schedulers/UnstructuredDetailedTasks.h>
#include <CCA/Components/Schedulers/UnstructuredDependencyBatch.h>
#include <CCA/Components/Schedulers/UnstructuredMemoryLog.h>
#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnstructuredSchedulerCommon.h>
#include <CCA/Components/Schedulers/UnstructuredTaskGraph.h>

#include <Core/Grid/UnstructuredGrid.h>
#include <Core/Grid/Variables/UnstructuredPSPatchMatlGhostRange.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>


#ifdef HAVE_CUDA
  #include <Core/Parallel/CrowdMonitor.hpp>
#endif

#include <sci_defs/cuda_defs.h>

#include <atomic>
#include <sstream>
#include <string>

using namespace Uintah;

namespace Uintah {
  // Used externally in UnstructuredDetailedTask.cc
  Uintah::MasterLock g_unstructured_external_ready_mutex{}; // synchronizes access to the external-ready task queue

  // used externally in UnstructuredDetailedTask.cc
  Dout g_unstructured_scrubbing_dbg(      "Scrubbing", "UnstructuredDetailedTasks", "report var scrubbing: see UnstructuredDetailedTasks.cc for usage", false);
  
  // used externally in UnstructuredDetailedTask.cc
  // for debugging - set the variable name (inside the quotes) and patchID to watch one in the scrubout
  std::string g_unstructured_var_scrub_dbg   = "";
  int         g_unstructured_patch_scrub_dbg = -1;
}

namespace {

  Uintah::MasterLock g_internal_ready_mutex{}; // synchronizes access to the internal-ready task queue
  
  Dout g_detailed_dw_dbg(    "UnstructuredDetailedDWDBG", "UnstructuredDetailedTasks", "report when var is saved in varDB", false);
  Dout g_detailed_tasks_dbg( "UnstructuredDetailedTasks", "UnstructuredDetailedTasks", "general bdg info for UnstructuredDetailedTasks", false);
  Dout g_message_tags_dbg(   "MessageTags",   "UnstructuredDetailedTasks", "info on MPI message tag assignment", false);


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


//_____________________________________________________________________________
//
UnstructuredDetailedTasks::UnstructuredDetailedTasks(       UnstructuredSchedulerCommon * sc
                            , const ProcessorGroup  * pg
                            , const UnstructuredTaskGraph       * taskgraph
                            , const std::set<int>   & neighborhood_processors
                            ,       bool              mustConsiderInternalDependencies /* = false */
                            )
  : m_sched_common(sc)
  , m_proc_group(pg)
  , m_task_graph(taskgraph)
  , m_must_consider_internal_deps(mustConsiderInternalDependencies)
{
  // Set up mappings for the initial send tasks
  int dwmap[UnstructuredTask::TotalDWs];
  for (int i = 0; i < UnstructuredTask::TotalDWs; i++) {
    dwmap[i] = UnstructuredTask::InvalidDW;
  }

  dwmap[UnstructuredTask::OldDW] = 0;
  dwmap[UnstructuredTask::NewDW] = UnstructuredTask::NoDW;

  m_send_old_data = scinew UnstructuredTask( "send old data", UnstructuredTask::InitialSend );
  m_send_old_data->m_phase = 0;
  m_send_old_data->setMapping( dwmap );

  // Create a send-old-data detailed task for every processor in my neighborhood.
  for (auto iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); ++iter) {
    UnstructuredDetailedTask* newtask = scinew UnstructuredDetailedTask( m_send_old_data, nullptr, nullptr, this );
    newtask->assignResource(*iter);

    // use a map because the processors in this map are likely to be sparse
    m_send_old_map[*iter] = m_tasks.size();
    m_tasks.push_back(newtask);
  }
}

//_____________________________________________________________________________
//
UnstructuredDetailedTasks::~UnstructuredDetailedTasks()
{
  // Free dynamically allocated SrubItems
  m_scrub_count_table.remove_all();

  for (size_t i = 0; i < m_dep_batches.size(); i++) {
    delete m_dep_batches[i];
  }

  for (size_t i = 0; i < m_tasks.size(); i++) {
    delete m_tasks[i];
  }

  delete m_send_old_data;
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::assignMessageTags( int me )
{
  // maps from, to (process) pairs to indices for each batch of that pair
  std::map<std::pair<int, int>, int> perPairBatchIndices;

  for (size_t i = 0; i < m_dep_batches.size(); i++) {
    UnstructuredDependencyBatch* batch = m_dep_batches[i];

    int from = batch->m_from_task->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, m_proc_group->nRanks());

    int to = batch->m_to_rank;
    ASSERTRANGE(to, 0, m_proc_group->nRanks());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      std::pair<int, int> fromToPair = std::make_pair(from, to);
      m_dep_batches[i]->m_message_tag = ++perPairBatchIndices[fromToPair];  // start with one
      DOUT(g_message_tags_dbg, "Rank-" << me << " assigning message tag " << batch->m_message_tag << " from task " << batch->m_from_task->getName()
                               << " to task " << batch->m_to_tasks.front()->getName() << ", rank-" << from << " to rank-" << to);
    }
  }

  if (g_message_tags_dbg) {
    for (auto iter = perPairBatchIndices.begin(); iter != perPairBatchIndices.end(); ++iter) {
      int from = iter->first.first;
      int to   = iter->first.second;
      int num  = iter->second;

      DOUT(true, num << " messages from rank-" << from << " to rank-" << to);;
    }
  }
}  // end assignMessageTags()

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::add( UnstructuredDetailedTask * dtask )
{
  m_tasks.push_back(dtask);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::makeDWKeyDatabase()
{
  for (auto i = 0u; i < m_local_tasks.size(); i++) {
    UnstructuredDetailedTask* task = m_local_tasks[i];
    //for reduction task check modifies other task check computes
    const UnstructuredTask::Dependency *comp = task->getTask()->isReductionUnstructuredTask() ? task->getTask()->getModifies() : task->getTask()->getComputes();
    for (; comp != nullptr; comp = comp->m_next) {
      const MaterialSubset* matls = comp->m_matls ? comp->m_matls : task->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredReductionVariable ||
            comp->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredSoleVariable) {
          m_level_keyDB.insert(comp->m_var, matl, comp->m_reduction_level);
        }
        else { // if variables saved on varDB
          const UnstructuredPatchSubset* patches = comp->m_patches ? comp->m_patches : task->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const UnstructuredPatch* patch = patches->get(p);
            m_var_keyDB.insert(comp->m_var, matl, patch);

            DOUT(g_detailed_dw_dbg, "reserve " << comp->m_var->getName() << " on UnstructuredPatch " << patch->getID() << ", Matl " << matl);
          }
        }
      } //end matls
    } // end comps
  } // end localtasks
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::computeLocalTasks( int me )
{
  if (m_local_tasks.size() != 0) {
    return;
  }

  int order = 0;
  m_initial_ready_tasks = TaskQueue();
  for (auto i = 0u; i < m_tasks.size(); i++) {
    UnstructuredDetailedTask* task = m_tasks[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, m_proc_group->nRanks());

    if (task->getAssignedResourceIndex() == me || task->getTask()->getType() == UnstructuredTask::Reduction) {
      m_local_tasks.push_back(task);

      if (task->areInternalDependenciesSatisfied()) {
        m_initial_ready_tasks.push(task);
      }
      task->assignStaticOrder(++order);
    }
  }
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::initializeScrubs( std::vector<UnstructuredOnDemandDataWarehouseP> & dws, int dwmap[] )
{
  DOUT(g_unstructured_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Begin initialize scrubs");

  std::vector<bool> initialized(dws.size(), false);
  for (int i = 0; i < (int)UnstructuredTask::TotalDWs; i++) {
    if (dwmap[i] < 0) {
      continue;
    }
    UnstructuredOnDemandDataWarehouse* dw = dws[dwmap[i]].get_rep();
    if (dw != nullptr && dw->getScrubMode() == UnstructuredDataWarehouse::ScrubComplete) {
      // only a OldDW or a CoarseOldDW will have scrubComplete 
      //   But we know a future taskgraph (in a w-cycle) will need the vars if there are fine dws 
      //   between New and Old.  In this case, the scrub count needs to be complemented with CoarseOldDW
      int tgtype = getTaskGraph()->getType();
      if (!initialized[dwmap[i]] || tgtype == UnstructuredScheduler::IntermediateTaskGraph) {
        // if we're intermediate, we're going to need to make sure we don't scrub CoarseOld before we finish using it
        DOUT(g_unstructured_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Initializing scrubs on dw: " << dw->getID()
                                      << " for DW type " << i << " ADD=" << initialized[dwmap[i]]);
        dw->initializeScrubs(i, &m_scrub_count_table, initialized[dwmap[i]]);
      }
      if (i != UnstructuredTask::OldDW && tgtype != UnstructuredScheduler::IntermediateTaskGraph && dwmap[UnstructuredTask::NewDW] - dwmap[UnstructuredTask::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        UnstructuredOnDemandDataWarehouse* olddw = dws[dwmap[UnstructuredTask::OldDW]].get_rep();
        DOUT(g_unstructured_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD=" << 1);
        ASSERT(initialized[dwmap[UnstructuredTask::OldDW]]);
        olddw->initializeScrubs(i, &m_scrub_count_table, true);
      }
      initialized[dwmap[i]] = true;
    }
  }

  DOUT(g_unstructured_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " End initialize scrubs");
}

//_____________________________________________________________________________
//
// used to be in terms of the dw index within the scheduler,
// but now store WhichDW.  This enables multiple tg execution
void
UnstructuredDetailedTasks::addScrubCount( const UnstructuredVarLabel * var
                            ,       int        matlindex
                            , const UnstructuredPatch    * patch
                            ,       int        dw
                            )
{
  if (patch->isVirtual()) {
    patch = patch->getRealUnstructuredPatch();
  }
  UnstructuredScrubItem key(var, matlindex, patch, dw);
  UnstructuredScrubItem* result;
  result = m_scrub_count_table.lookup(&key);
  if (!result) {
    result = scinew UnstructuredScrubItem(var, matlindex, patch, dw);
    m_scrub_count_table.insert(result);
  }
  result->m_count++;
  if (g_unstructured_scrubbing_dbg && (var->getName() == g_unstructured_var_scrub_dbg || g_unstructured_var_scrub_dbg == "") && (g_unstructured_patch_scrub_dbg == patch->getID() || g_unstructured_patch_scrub_dbg == -1)) {
    DOUT(true, "Rank-" << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/"
                       << patch->getID() << "/" << matlindex << "/" << *var << ": " << result->m_count);
  }
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::setScrubCount( const UnstructuredTask::Dependency                    * req
                            ,       int                                   matl
                            , const UnstructuredPatch                               * patch
                            ,       std::vector<UnstructuredOnDemandDataWarehouseP> & dws
                            )
{
  ASSERT(!patch->isVirtual());
  UnstructuredDataWarehouse::ScrubMode scrubmode = dws[req->mapDataWarehouse()]->getScrubMode();
  const std::set<const UnstructuredVarLabel*, UnstructuredVarLabel::Compare>& initialRequires = getSchedulerCommon()->getInitialRequiredVars();
  if (scrubmode == UnstructuredDataWarehouse::ScrubComplete || (scrubmode == UnstructuredDataWarehouse::ScrubNonPermanent
      && initialRequires.find(req->m_var) == initialRequires.end())) {
    int scrubcount;
    if (!getScrubCount(req->m_var, matl, patch, req->m_whichdw, scrubcount)) {
      SCI_THROW(InternalError("No scrub count for received MPIVariable: "+req->m_var->getName(), __FILE__, __LINE__));
    }

    if (g_unstructured_scrubbing_dbg && (req->m_var->getName() == g_unstructured_var_scrub_dbg || g_unstructured_var_scrub_dbg == "") && (g_unstructured_patch_scrub_dbg == patch->getID() || g_unstructured_patch_scrub_dbg == -1)) {
      DOUT(true, "Rank-" << Parallel::getMPIRank() << " setting scrubcount for recv of " << req->mapDataWarehouse() << "/" << patch->getID()
                         << "/" << matl << "/" << req->m_var->getName() << ": " << scrubcount);
    }
    dws[req->mapDataWarehouse()]->setScrubCount(req->m_var, matl, patch, scrubcount);
  }
}

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getScrubCount( const UnstructuredVarLabel * label
                            ,       int        matlIndex
                            , const UnstructuredPatch    * patch
                            ,       int        dw
                            ,       int&       count
                            )
{
  ASSERT(!patch->isVirtual());
  UnstructuredScrubItem key(label, matlIndex, patch, dw);
  UnstructuredScrubItem* result = m_scrub_count_table.lookup(&key);
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
UnstructuredDetailedTasks::createScrubCounts()
{
  // Clear old ScrubItems
  m_scrub_count_table.remove_all();

  // Go through each of the tasks and determine which variables it will require
  for (int i = 0; i < (int)m_local_tasks.size(); i++) {
    UnstructuredDetailedTask* dtask = m_local_tasks[i];
    const UnstructuredTask* task = dtask->getTask();
    for (const UnstructuredTask::Dependency* req = task->getRequires(); req != nullptr; req = req->m_next) {
      constHandle<UnstructuredPatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->m_whichdw;
      UnstructuredTypeDescription::UnstructuredType type = req->m_var->typeDescription()->getUnstructuredType();
      if (type != UnstructuredTypeDescription::UnstructuredReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const UnstructuredPatch* patch = patches->get(i);
          UnstructuredPatch::selectType neighbors;
          IntVector low, high;
          patch->computeVariableExtents(type, req->m_var->getBoundaryLayer(), req->m_gtype, req->m_num_ghost_cells, neighbors, low, high);
          for (unsigned int i = 0; i < neighbors.size(); i++) {
            const UnstructuredPatch* neighbor = neighbors[i];
            for (int m = 0; m < matls->size(); m++) {
              addScrubCount(req->m_var, matls->get(m), neighbor, whichdw);
            }
          }
        }
      }
    }

    // determine which variables this task will modify
    for (const UnstructuredTask::Dependency* req = task->getModifies(); req != nullptr; req = req->m_next) {
      constHandle<UnstructuredPatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->m_whichdw;
      UnstructuredTypeDescription::UnstructuredType type = req->m_var->typeDescription()->getUnstructuredType();
      if (type != UnstructuredTypeDescription::UnstructuredReductionVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const UnstructuredPatch* patch = patches->get(i);
          for (int m = 0; m < matls->size(); m++) {
            addScrubCount(req->m_var, matls->get(m), patch, whichdw);
          }
        }
      }
    }
  }

  if (g_unstructured_scrubbing_dbg) {
    std::ostringstream message;
    message << Parallel::getMPIRank() << " scrub counts:\n";
    message << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for (FastHashTableIter<UnstructuredScrubItem> iter(&m_scrub_count_table); iter.ok(); ++iter) {
      const UnstructuredScrubItem* rec = iter.get_key();
      message << rec->m_dw << '/' << (rec->m_patch ? rec->m_patch->getID() : 0) << '/' << rec->m_matl << '/' << rec->m_label->getName()
               << "\t\t" << rec->m_count << '\n';
    }
    message << "Rank-" << Parallel::getMPIRank() << " end scrub counts";
    DOUT(true, message.str());
  }
}

//_____________________________________________________________________________
//
UnstructuredDetailedDep*
UnstructuredDetailedTasks::findMatchingDetailedDep(UnstructuredDependencyBatch  * batch
                                      ,       UnstructuredDetailedTask     * toTask
                                      ,       UnstructuredTask::Dependency * req
                                      , const UnstructuredPatch            * fromPatch
                                      ,       int                matl
                                      ,       IntVector          low
                                      ,       IntVector          high
                                      ,       IntVector        & totalLow
                                      ,       IntVector        & totalHigh
                                      ,       UnstructuredDetailedDep     *& parent_dep
                                      )
{
  totalLow = low;
  totalHigh = high;
  UnstructuredDetailedDep* dep = batch->m_head;

  parent_dep = nullptr;
  UnstructuredDetailedDep* last_dep = nullptr;
  UnstructuredDetailedDep* valid_dep = nullptr;

  //search each dep
  for (; dep != nullptr; dep = dep->m_next) {
    //if deps are equivalent
    if (fromPatch == dep->m_from_patch &&  matl == dep->m_matl && (req == dep->m_req || (req->m_var->equals(dep->m_req->m_var) && req->mapDataWarehouse() == dep->m_req->mapDataWarehouse()))) {

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

      if (m_sched_common->useSmallMessages()) {
        // If two patches on the same processor want data from the same patch on a different
        // processor, we can either pack them in one dependency and send the min and max of their range (which
        // will frequently result in sending the entire patch), or we can use two dependencies (which will get packed into
        // one message) and only send the resulting data.

        // We want to create a new dep in such cases.
        // However, we don't want to do this in cases where we simply add more ghost cells.
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
            if (valid_dep == nullptr) {
              parent_dep = last_dep;
              valid_dep = dep;
            }
            //keep searching in case there is a better dep to combine with
          }
        }
        else if (g_detailed_tasks_dbg) {
          std::ostringstream message;
          message << m_proc_group->myRank() << "            Ignoring: " << dep->m_low << " " << dep->m_high << ", fromPatch = ";
          if (fromPatch) {
            message << fromPatch->getID() << '\n';
          }
          else {
            message << "nullptr\n";
          }
          message << m_proc_group->myRank() << " TP: " << totalLow << " " << totalHigh;
          DOUT(true, message.str());
        }
      }
      else {
        if (extraComm) {
          m_extra_comm += newSize - (requiredSize + oldSize);
        }
        //not using small messages so take the first dep you find, it will be extended
        valid_dep = dep;
        break;
      }
    }
    //pointer to dependency before this dep so insertion/deletion can be done quicker
    last_dep = dep;
  }
  if (valid_dep == nullptr) {
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
 * at the same location that it was first combined.  This is to ensure all future dependencies
 * combine with the same dependencies as the original.
 */
void
UnstructuredDetailedTasks::possiblyCreateDependency(       UnstructuredDetailedTask     * from
                                       ,       UnstructuredTask::Dependency * comp
                                       , const UnstructuredPatch            * fromPatch
                                       ,       UnstructuredDetailedTask     * to
                                       ,       UnstructuredTask::Dependency * req
                                       , const UnstructuredPatch            * toPatch
                                       ,       int                matl
                                       , const IntVector        & low
                                       , const IntVector        & high
                                       ,       DepCommCond        cond
                                       )
{
  ASSERTRANGE(from->getAssignedResourceIndex(), 0, m_proc_group->nRanks());
  ASSERTRANGE(to->getAssignedResourceIndex(),   0, m_proc_group->nRanks());

  int my_rank = m_proc_group->myRank();

  if (g_detailed_tasks_dbg) {
    std::ostringstream message;
    message << "Rank-" << my_rank << "  " << *to << " depends on " << *from << "\n";
      if (comp) {
        message << "Rank-" << my_rank << "  From comp " << *comp;
      }
      else {
        message << "Rank-" << my_rank << "  From OldDW ";
      }
      message << " to req " << *req;
      DOUT(true, message.str());
  }

  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();

  // if neither task talks to this processor, return
  if (fromresource != my_rank && toresource != my_rank) {
    return;
  }

  if ((toresource == my_rank || (req->m_patches_dom != UnstructuredTask::ThisLevel && fromresource == my_rank))
      && fromPatch && !req->m_var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->m_var, matl, fromPatch, req->m_whichdw);
  }

  // if the dependency is on the same processor then add an internal dependency
  if (fromresource == my_rank && fromresource == toresource) {
    to->addInternalDependency(from, req->m_var);

    // In case of multiple GPUs per node, we don't return.  Multiple GPUs
    // need internal dependencies to communicate data.
    if ( ! Uintah::Parallel::usingDevice()) {
      return;
    }

  }

  // this should have been pruned out earlier
  ASSERT(!req->m_var->typeDescription()->isReductionVariable())
  // Do not check external deps on UnstructuredSoleVariable
  if (req->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredSoleVariable) {
    return;
  }

#ifdef HAVE_CUDA
  if (Uintah::Parallel::usingDevice()) {
    if (fromresource == my_rank && fromresource == toresource) {
      if (fromPatch != toPatch) {
        //printf("In UnstructuredDetailedTasks::createInternalDependencyBatch creating internal dependency from patch %d to patch %d, from task %p to task %p\n", fromPatch->getID(), toPatch->getID(), from, to);
        createInternalDependencyBatch(from, comp, fromPatch, to, req, toPatch, matl, low, high, cond);
      }
      return; //We've got all internal dependency information for the GPU, now we can return.
    }
  }
#endif

  // make keys for MPI messages
  if (fromPatch) {
    m_var_keyDB.insert(req->m_var, matl, fromPatch);
  }

  // get dependency batch
  UnstructuredDependencyBatch* batch = from->getComputes();

  // find dependency batch that is to the same processor as this dependency
  for (; batch != nullptr; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource) {
      break;
    }
  }

  // if batch doesn't exist then create it
  if (!batch) {
    batch = scinew UnstructuredDependencyBatch(toresource, from, to);
    m_dep_batches.push_back(batch);
    from->addComputes(batch);

#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addRequires(batch);
#else
    to->addRequires(batch);
#endif

    ASSERTL2(newRequireBatch);

    DOUT(g_detailed_tasks_dbg, "Rank-" << my_rank << "          NEW BATCH!");
  }
  else if (m_must_consider_internal_deps) {  // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add to the batch's toTasks.
      batch->m_to_tasks.push_back(to);
    }
    DOUT(g_detailed_tasks_dbg, "Rank-" << my_rank << "          USING PREVIOUSLY CREATED BATCH!");
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  // create the new dependency
  UnstructuredDetailedDep* new_dep = scinew UnstructuredDetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  // search for a dependency that can be combined with this dependency

  // location of parent dependency
  UnstructuredDetailedDep* parent_dep;

  UnstructuredDetailedDep* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl,
                                                      new_dep->m_low, new_dep->m_high,
                                                      varRangeLow, varRangeHigh, parent_dep);

  // This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  UnstructuredDetailedDep* insert_dep = parent_dep;

  // if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  while (matching_dep != nullptr) {

    // debugging output
    if (g_detailed_tasks_dbg) {
      std::ostringstream message;
      message << "Rank-" << my_rank << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
              << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      message << *req->m_var << '\n';
      message << *new_dep->m_req->m_var << '\n';
      if (comp) {
        message << *comp->m_var << '\n';
      }
      if (new_dep->m_comp) {
        message << *new_dep->m_comp->m_var;
      }
      DOUT(true, message.str());
    }

    // extend the dependency range
    new_dep->m_low  = Min(new_dep->m_low,  matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);

    // TODO This has broken OutputNthProc - figure this out ASAP and fix, APH 02/23/18
//    // if the same dependency already exists then short circuit out of this function.
//    if (matching_dep->m_low == new_dep->m_low && matching_dep->m_high == new_dep->m_high) {
//      matching_dep->m_to_tasks.splice(matching_dep->m_to_tasks.begin(), new_dep->m_to_tasks);
//      delete new_dep;
//      return;
//    }

    // copy matching dependencies toTasks to the new dependency
    new_dep->m_to_tasks.splice(new_dep->m_to_tasks.begin(), matching_dep->m_to_tasks);

    // erase particle sends/recvs
    if (req->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredParticleVariable && req->m_whichdw == UnstructuredTask::OldDW) {
      UnstructuredPSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x") {
        DOUT(g_detailed_tasks_dbg, "Rank-" << my_rank << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
                                           << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
                                           << " cond " << cond << " dw " << req->mapDataWarehouse());
      }

      if (fromresource == my_rank) {
        std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter != m_particle_sends[toresource].end());

        //subtract one from the count
        iter->count_--;

        // if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
        }
      }
      else if (toresource == my_rank) {
        std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter != m_particle_recvs[fromresource].end());
        // subtract one from the count
        iter->count_--;

        // if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          m_particle_recvs[fromresource].erase(iter);
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
  new_dep->m_patch_low  = varRangeLow;
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
  if (req->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredParticleVariable && req->m_whichdw == UnstructuredTask::OldDW) {
    UnstructuredPSPatchMatlGhostRange pmg = UnstructuredPSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == my_rank) {
      std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
      if (iter == m_particle_sends[toresource].end()) {  //if does not exist
        //add to the sends list
        m_particle_sends[toresource].insert(pmg);
      }
      else {
        // increment count
        iter->count_++;
      }
    }
    else if (toresource == my_rank) {
      std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        //add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      }
      else {
        //increment the count
        iter->count_++;
      }

    }
    if (req->m_var->getName() == "p.x") {
      DOUT(g_detailed_tasks_dbg, "Rank-" << my_rank << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
                                         << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
                                         << req->mapDataWarehouse());
    }
  }

  if (g_detailed_tasks_dbg) {
    std::ostringstream message;
    message << "Rank-" << my_rank << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch) {
      message << fromPatch->getID() << '\n';
    }
    else {
      message << "nullptr";
    }
    DOUT(true, message.str());
  }
}

//_____________________________________________________________________________
//
UnstructuredDetailedTask*
UnstructuredDetailedTasks::getOldDWSendTask( int proc )
{
#if SCI_ASSERTION_LEVEL>0
  // verify the map entry has been created
  if (m_send_old_map.find(proc) == m_send_old_map.end()) {
    std::cout << m_proc_group->myRank() << " Error trying to get oldDWSendTask for processor: " << proc << " but it does not exist\n";
    throw InternalError("oldDWSendTask does not exist", __FILE__, __LINE__);
  }
#endif 
  return m_tasks[m_send_old_map[proc]];
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::internalDependenciesSatisfied( UnstructuredDetailedTask * dtask )
{
  std::lock_guard<Uintah::MasterLock> internal_ready_guard(g_internal_ready_mutex);

  m_ready_tasks.push(dtask);
  atomic_readyTasks_size.fetch_add(1, std::memory_order_relaxed);
}

//_____________________________________________________________________________
//
UnstructuredDetailedTask*
UnstructuredDetailedTasks::getNextInternalReadyTask()
{


  UnstructuredDetailedTask* nextTask = nullptr;
  if (atomic_readyTasks_size.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<Uintah::MasterLock> internal_ready_guard(g_internal_ready_mutex);
    if (!m_ready_tasks.empty()) {

      nextTask = m_ready_tasks.front();
      atomic_readyTasks_size.fetch_sub(1, std::memory_order_relaxed);
      m_ready_tasks.pop();
    }
  }

  return nextTask;
}

//_____________________________________________________________________________
//
int
UnstructuredDetailedTasks::numInternalReadyTasks()
{
  //std::lock_guard<Uintah::MasterLock> internal_ready_guard(g_internal_ready_mutex);
  //return readyTasks_.size();
  return atomic_readyTasks_size.load(std::memory_order_relaxed);
}

//_____________________________________________________________________________
//
UnstructuredDetailedTask*
UnstructuredDetailedTasks::getNextExternalReadyTask()
{


  UnstructuredDetailedTask* nextTask = nullptr;
  if (m_atomic_mpi_completed_tasks_size.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<Uintah::MasterLock> external_ready_guard(g_unstructured_external_ready_mutex);
    if (!m_mpi_completed_tasks.empty()) {
      nextTask = m_mpi_completed_tasks.top();
      m_atomic_mpi_completed_tasks_size.fetch_sub(1, std::memory_order_relaxed);
      m_mpi_completed_tasks.pop();
    }
  }

  return nextTask;
}

//_____________________________________________________________________________
//
int
UnstructuredDetailedTasks::numExternalReadyTasks()
{
  //std::lock_guard<Uintah::MasterLock> external_ready_guard(g_unstructured_external_ready_mutex);
  //return mpiCompletedTasks_.size();
  return m_atomic_mpi_completed_tasks_size.load(std::memory_order_relaxed);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::initTimestep()
{
  m_ready_tasks = m_initial_ready_tasks;
  atomic_readyTasks_size.store(m_initial_ready_tasks.size(), std::memory_order_relaxed);
  incrementDependencyGeneration();
  initializeBatches();
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::incrementDependencyGeneration()
{
  if (m_current_dependency_generation >= ULONG_MAX) {
    SCI_THROW(InternalError("UnstructuredDetailedTasks::currentDependencySatisfyingGeneration has overflowed", __FILE__, __LINE__));
  }
  m_current_dependency_generation++;
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::initializeBatches()
{
  for (int i = 0; i < static_cast<int>(m_dep_batches.size()); i++) {
    m_dep_batches[i]->reset();
  }
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::logMemoryUse(       std::ostream  & out
                           ,       unsigned long & total
                           , const std::string   & tag
                           )
{
  std::ostringstream elems1;
  elems1 << m_tasks.size();
  logMemory(out, total, tag, "tasks", "UnstructuredDetailedTask", nullptr, -1, elems1.str(), m_tasks.size() * sizeof(UnstructuredDetailedTask), 0);
  std::ostringstream elems2;
  elems2 << m_dep_batches.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", nullptr, -1, elems2.str(), m_dep_batches.size() * sizeof(UnstructuredDependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    for (UnstructuredDetailedDep* dep = m_dep_batches[i]->m_head; dep != nullptr; dep = dep->m_next) {
      ndeps++;
    }
  }
  std::ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "UnstructuredDetailedDep", nullptr, -1, elems3.str(), ndeps * sizeof(UnstructuredDetailedDep), 0);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::emitEdges( ProblemSpecP edgesElement, int rank )
{
  for (int i = 0; i < static_cast<int>(m_tasks.size()); i++) {
    ASSERTRANGE(m_tasks[i]->getAssignedResourceIndex(), 0, m_proc_group->nRanks());
    if (m_tasks[i]->getAssignedResourceIndex() == rank) {
      m_tasks[i]->emitEdges(edgesElement);
    }
  }
}

//_____________________________________________________________________________
//
// comparing the priority of two detailed tasks - true means give rtask priority
bool
UnstructuredDetailedTaskPriorityComparison::operator()( UnstructuredDetailedTask *& ltask
                                          , UnstructuredDetailedTask *& rtask
                                          )
{
  QueueAlg alg = ltask->getTaskGroup()->getTaskPriorityAlg();
  ASSERT(alg == rtask->getTaskGroup()->getTaskPriorityAlg());

  if (alg == FCFS) {
    return false;               // First Come First Serve;
  }

  if (alg == Stack) {
    return true;               // First Come Last Serve, for robust testing;
  }

  if (alg == Random) {
    return (random() % 2 == 0);   // Random;
  }

  if (ltask->getTask()->getSortedOrder() > rtask->getTask()->getSortedOrder()) {
    return true;
  }

  if (ltask->getTask()->getSortedOrder() < rtask->getTask()->getSortedOrder()) {
    return false;
  }

  else if (alg == MostMessages || alg == LeastMessages) {
    int lmsg = 0;
    int rmsg = 0;
    for (UnstructuredDependencyBatch* batch = ltask->getComputes(); batch != nullptr; batch = batch->m_comp_next) {
      for (UnstructuredDetailedDep* dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
        lmsg++;
      }
    }
    for (UnstructuredDependencyBatch* batch = rtask->getComputes(); batch != nullptr; batch = batch->m_comp_next) {
      for (UnstructuredDetailedDep* dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
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
  else if (alg == PatchOrder) {  // smaller level, larger size, smaller patchID, smaller tasksortID
    const UnstructuredPatchSubset* lpatches = ltask->getPatches();
    const UnstructuredPatchSubset* rpatches = rtask->getPatches();
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
  else if (alg == PatchOrderRandom) {  // smaller level, larger size, smaller patchID, smaller tasksortID
    const UnstructuredPatchSubset* lpatches = ltask->getPatches();
    const UnstructuredPatchSubset* rpatches = rtask->getPatches();
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



#ifdef HAVE_CUDA

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDeviceValidateRequiresCopiesTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_validateRequiresCopies_pool_iter = device_validateRequiresCopies_pool.find_any(ready_request);

  if (device_validateRequiresCopies_pool_iter) {
    dtask = *device_validateRequiresCopies_pool_iter;
    device_validateRequiresCopies_pool.erase(device_validateRequiresCopies_pool_iter);
    //printf("device_validateRequiresCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_validateRequiresCopies_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDevicePerformGhostCopiesTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_performGhostCopies_pool_iter = device_performGhostCopies_pool.find_any(ready_request);

  if (device_performGhostCopies_pool_iter) {
    dtask = *device_performGhostCopies_pool_iter;
    device_performGhostCopies_pool.erase(device_performGhostCopies_pool_iter);
    //printf("device_performGhostCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_performGhostCopies_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDeviceValidateGhostCopiesTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_validateGhostCopies_pool_iter = device_validateGhostCopies_pool.find_any(ready_request);

  if (device_validateGhostCopies_pool_iter) {
    dtask = *device_validateGhostCopies_pool_iter;
    device_validateGhostCopies_pool.erase(device_validateGhostCopies_pool_iter);
    //printf("device_validateGhostCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_validateGhostCopies_pool.size());
    retVal = true;
  }

  return retVal;
}


//______________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDeviceCheckIfExecutableTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_checkIfExecutable_pool_iter = device_checkIfExecutable_pool.find_any(ready_request);
  if (device_checkIfExecutable_pool_iter) {
    dtask = *device_checkIfExecutable_pool_iter;
    device_checkIfExecutable_pool.erase(device_checkIfExecutable_pool_iter);
    //printf("device_checkIfExecutable_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_checkIfExecutable_pool.size());
    retVal = true;
  }

  return retVal;
}

//______________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDeviceReadyToExecuteTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_readyToExecute_pool_iter = device_readyToExecute_pool.find_any(ready_request);
  if (device_readyToExecute_pool_iter) {
    dtask = *device_readyToExecute_pool_iter;
    device_readyToExecute_pool.erase(device_readyToExecute_pool_iter);
    //printf("device_readyToExecute_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_readyToExecute_pool.size());
    retVal = true;
  }

  return retVal;
}


//______________________________________________________________________
//
bool
UnstructuredDetailedTasks::getDeviceExecutionPendingTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_executionPending_pool_iter = device_executionPending_pool.find_any(ready_request);
  if (device_executionPending_pool_iter) {
    dtask = *device_executionPending_pool_iter;
    device_executionPending_pool.erase(device_executionPending_pool_iter);
    //printf("device_executionPending_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_executionPending_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getHostValidateRequiresCopiesTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator host_validateRequiresCopies_pool_iter = host_validateRequiresCopies_pool.find_any(ready_request);

  if (host_validateRequiresCopies_pool_iter) {
    dtask = *host_validateRequiresCopies_pool_iter;
    host_validateRequiresCopies_pool.erase(host_validateRequiresCopies_pool_iter);
    //printf("host_validateRequiresCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), host_validateRequiresCopies_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
UnstructuredDetailedTasks::getHostCheckIfExecutableTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator host_checkIfExecutable_pool_iter = host_checkIfExecutable_pool.find_any(ready_request);
  if (host_checkIfExecutable_pool_iter) {
    dtask = *host_checkIfExecutable_pool_iter;
    host_checkIfExecutable_pool.erase(host_checkIfExecutable_pool_iter);
    //printf("host_checkIfExecutable_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), host_checkIfExecutable_pool.size());
    retVal = true;
  }

  return retVal;
}

//______________________________________________________________________
//
bool
UnstructuredDetailedTasks::getHostReadyToExecuteTask(UnstructuredDetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](UnstructuredDetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator host_readyToExecute_pool_iter = host_readyToExecute_pool.find_any(ready_request);
  if (host_readyToExecute_pool_iter) {
    dtask = *host_readyToExecute_pool_iter;
    host_readyToExecute_pool.erase(host_readyToExecute_pool_iter);
    //printf("host_readyToExecute_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), host_readyToExecute_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addDeviceValidateRequiresCopies(UnstructuredDetailedTask * dtask)
{
  device_validateRequiresCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addDevicePerformGhostCopies(UnstructuredDetailedTask * dtask)
{
  device_performGhostCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addDeviceValidateGhostCopies(UnstructuredDetailedTask * dtask)
{
  device_validateGhostCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addDeviceCheckIfExecutable(UnstructuredDetailedTask * dtask)
{
  device_checkIfExecutable_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::addDeviceReadyToExecute( UnstructuredDetailedTask * dtask )
{
  device_readyToExecute_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::addDeviceExecutionPending( UnstructuredDetailedTask * dtask )
{
  device_executionPending_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addHostValidateRequiresCopies(UnstructuredDetailedTask * dtask)
{
  host_validateRequiresCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::addHostCheckIfExecutable(UnstructuredDetailedTask * dtask)
{
  host_checkIfExecutable_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
UnstructuredDetailedTasks::addHostReadyToExecute( UnstructuredDetailedTask * dtask )
{
  host_readyToExecute_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void UnstructuredDetailedTasks::createInternalDependencyBatch(       UnstructuredDetailedTask     * from
                                                 ,       UnstructuredTask::Dependency * comp
                                                 , const UnstructuredPatch            * fromPatch
                                                 ,       UnstructuredDetailedTask     * to
                                                 ,       UnstructuredTask::Dependency * req
                                                 , const UnstructuredPatch            * toPatch
                                                 ,       int                matl
                                                 , const IntVector        & low
                                                 , const IntVector        & high
                                                 ,       DepCommCond        cond
                                                 )
{

  // get dependency batch
  DependencyBatch* batch = from->getInternalComputes();
  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();
  // find dependency batch that is to the same processor as this dependency
  for (; batch != nullptr; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource)
      break;
  }

  // if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, to);
    m_dep_batches.push_back(batch);  // Should be fine to push this batch on here, at worst
                                // MPI message tags are created for these which won't get used.
    from->addInternalComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addInternalRequires(batch);
#else
    to->addInternalRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if (g_detailed_tasks_dbg)
      DOUT(true, "Rank-" << m_proc_group->myRank() << "          NEW BATCH!");
  } else if (m_must_consider_internal_deps) {  // i.e. threaded mode
    if (to->addInternalRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->m_to_tasks.push_back(to);
    }

    DOUT(g_detailed_tasks_dbg, "Rank-" << m_proc_group->myRank() << "          USING PREVIOUSLY CREATED BATCH!");
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  //create the new dependency
  UnstructuredDetailedDep* new_dep = scinew UnstructuredDetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  UnstructuredDetailedDep* parent_dep;

  UnstructuredDetailedDep* matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high,
                                                              varRangeLow, varRangeHigh, parent_dep);

  //This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  UnstructuredDetailedDep* insert_dep = parent_dep;
  //If two dependencies are going to be on two different GPUs,
  //then do not merge the two dependencies into one collection.
  //Instead, keep them separate.  If they will be on the same GPU
  //then we can merge the dependencies as normal.
  //This means that no


  //if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  //check if we should combine them.  If the  new_dep->low
  while (matching_dep != nullptr) {

    //debugging output
    if (g_detailed_tasks_dbg) {
      std::ostringstream message;
      message << m_proc_group->myRank() << "            EXTENDED from " << new_dep->m_low << " " << new_dep->m_high << " to "
              << Min(new_dep->m_low, matching_dep->m_low) << " " << Max(new_dep->m_high, matching_dep->m_high) << "\n";
      message << *req->m_var << '\n';
      message << *new_dep->m_req->m_var << '\n';
      if (comp) {
        message << *comp->m_var << '\n';
      }
      if (new_dep->m_comp) {
        message << *new_dep->m_comp->m_var << '\n';
      }
      DOUT(true, message.str());
    }



    //extend the dependency range
    new_dep->m_low = Min(new_dep->m_low, matching_dep->m_low);
    new_dep->m_high = Max(new_dep->m_high, matching_dep->m_high);

    //copy matching dependencies toTasks to the new dependency
    new_dep->m_to_tasks.splice(new_dep->m_to_tasks.begin(), matching_dep->m_to_tasks);

    //erase particle sends/recvs
    if (req->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::ParticleVariable && req->m_whichdw == UnstructuredTask::OldDW) {
      UnstructuredPSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x")
        DOUT(g_detailed_tasks_dbg, "Rank-" << m_proc_group->myRank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
                          << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
                          << " cond " << cond << " dw " << req->mapDataWarehouse());

      if (fromresource == m_proc_group->myRank()) {
        std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter!=m_particle_sends[toresource].end());

        //subtract one from the count
        iter->count_--;

        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
        }
      } else if (toresource == m_proc_group->myRank()) {
        std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
        ASSERT(iter!=m_particle_recvs[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          m_particle_recvs[fromresource].erase(iter);
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
  new_dep->m_patch_low  = varRangeLow;
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
  if (req->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::ParticleVariable && req->m_whichdw == UnstructuredTask::OldDW) {
    UnstructuredPSPatchMatlGhostRange pmg = UnstructuredPSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == m_proc_group->myRank()) {
      std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
      if (iter == m_particle_sends[toresource].end())  //if does not exist
          {
        //add to the sends list
        m_particle_sends[toresource].insert(pmg);
      } else {
        //increment count
        iter->count_++;
      }
    } else if (toresource == m_proc_group->myRank()) {
      std::set<UnstructuredPSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
      if (iter == m_particle_recvs[fromresource].end()) {
        // add to the recvs list
        m_particle_recvs[fromresource].insert(pmg);
      } else {
        // increment the count
        iter->count_++;
      }

    }
    if (req->m_var->getName() == "p.x")
      DOUT(g_detailed_tasks_dbg, "Rank-" << m_proc_group->myRank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
                                         << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
                                         << req->mapDataWarehouse());
  }

  if (g_detailed_tasks_dbg) {
    std::ostringstream message;
    message << m_proc_group->myRank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch) {
      message << fromPatch->getID() << '\n';
    }
    else {
      message << "nullptr";
    }
    DOUT(true, message.str());
  }

}

//_____________________________________________________________________________
//
UnstructuredDetailedDep* UnstructuredDetailedTasks::findMatchingInternalDetailedDep(DependencyBatch         * batch
                                                           ,       UnstructuredDetailedTask     * toTask
                                                           ,       UnstructuredTask::Dependency * req
                                                           , const UnstructuredPatch            * fromPatch
                                                           ,       int                matl
                                                           ,       IntVector          low
                                                           ,       IntVector          high
                                                           ,       IntVector        & totalLow
                                                           ,       IntVector        & totalHigh
                                                           ,       UnstructuredDetailedDep     *& parent_dep
                                                           )
{
  totalLow   = low;
  totalHigh  = high;
  parent_dep = nullptr;

  UnstructuredDetailedDep * dep       = batch->m_head;
  UnstructuredDetailedDep * last_dep  = nullptr;
  UnstructuredDetailedDep * valid_dep = nullptr;

  //For now, turning off a feature that can combine ghost cells into larger vars
  //for scenarios where one source ghost cell var can handle more than one destination patch.

  //search each dep
  for (; dep != nullptr; dep = dep->m_next) {
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

  if (valid_dep == nullptr) {
    parent_dep = last_dep;
  }

  return valid_dep;
}

#endif


