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

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/DependencyBatch.h>
#include <CCA/Components/Schedulers/MemoryLog.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/visit_defs.h>

#ifdef HAVE_CUDA
  #include <Core/Parallel/CrowdMonitor.hpp>
#endif

#include <atomic>
#include <sstream>
#include <string>

using namespace Uintah;

namespace Uintah {
  // Used externally in DetailedTask.cc
  Uintah::MasterLock g_external_ready_mutex{}; // synchronizes access to the external-ready task queue

  // used externally in DetailedTask.cc
  Dout g_scrubbing_dbg(      "Scrubbing", "DetailedTasks", "report var scrubbing: see DetailedTasks.cc for usage", false);
  
  // used externally in DetailedTask.cc
  // for debugging - set the variable name (inside the quotes) and patchID to watch one in the scrubout
  std::string g_var_scrub_dbg   = "";
  int         g_patch_scrub_dbg = -1;
}

namespace {

  Uintah::MasterLock g_internal_ready_mutex{}; // synchronizes access to the internal-ready task queue
  
  Dout g_detailed_dw_dbg(    "DetailedDWDBG", "DetailedTasks", "report when var is saved in varDB", false);
  Dout g_detailed_tasks_dbg( "DetailedTasks", "DetailedTasks", "general bdg info for DetailedTasks", false);
  Dout g_message_tags_dbg(           "MessageTags",         "DetailedTasks", "info on MPI message tag assignment", false);
  Dout g_message_tags_stats_dbg(     "MessageTagStats",     "DetailedTasks", "stats on MPI message tag assignment", false);
#ifdef HAVE_VISIT
  Dout g_message_tags_task_stats_dbg("MessageTagTaskStats", "DetailedTasks", "stats on MPI message tag task assignment", false);
#endif  

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
DetailedTasks::DetailedTasks(       SchedulerCommon         * sc
                            , const ProcessorGroup          * pg
                            , const TaskGraph               * taskgraph
                            , const std::unordered_set<int> & neighborhood_processors
                            ,       bool                      mustConsiderInternalDependencies /* = false */
                            )
  : m_sched_common{sc}
  , m_proc_group{pg}
  , m_task_graph{taskgraph}
  , m_must_consider_internal_deps{mustConsiderInternalDependencies}
{
  // Set up mappings for the initial send tasks
  int dwmap[Task::TotalDWs];
  for (int i = 0; i < Task::TotalDWs; i++) {
    dwmap[i] = Task::InvalidDW;
  }

  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = Task::NoDW;

  m_send_old_data = scinew Task( "send_old_data", Task::InitialSend );
  m_send_old_data->m_phase = 0;
  m_send_old_data->setMapping( dwmap );

  // Create a send-old-data detailed task for every processor in my neighborhood.
  for (auto iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); ++iter) {
    DetailedTask* newtask = scinew DetailedTask( m_send_old_data, nullptr, nullptr, this );
    newtask->assignResource(*iter);

    // use a map because the processors in this map are likely to be sparse
    m_send_old_map[*iter] = m_tasks.size();
    m_tasks.push_back(newtask);
  }
}

//_____________________________________________________________________________
//
DetailedTasks::~DetailedTasks()
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
DetailedTasks::assignMessageTags( unsigned int index )
{
  int me     = m_proc_group->myRank();
  int nRanks = m_proc_group->nRanks();

  std::pair< std::string, std::string > allTasks("All", "Tasks");

  m_comm_info.clear();

  // Insert the stats to be collected for all tasks. In this case
  // collect stats for all tasks on this rank only for both messages
  // passed to it and recieved from other ranks.

  // These counts which start at one serve as the message tags.
  m_comm_info[ allTasks ].setKeyName( "Rank" );
  m_comm_info[ allTasks ].insert( CommPTPMsgTo,   std::string("ToRank")  , "messages" );
  m_comm_info[ allTasks ].insert( CommPTPMsgFrom, std::string("FromRank"), "messages" );
    
  // Map for individual task pairs.
  std::map< std::pair< std::string, std::string >, int > taskPairs;

  // Loop through all of the tasks to get the counts.
  for (size_t i = 0; i < m_dep_batches.size(); i++) {
    DependencyBatch* batch = m_dep_batches[i];

    int from = batch->m_from_task->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, m_proc_group->nRanks());

    int to = batch->m_to_rank;
    ASSERTRANGE(to, 0, m_proc_group->nRanks());

    // Only look at tasks assigned to this rank.
    if (from == me || to == me) {

#ifdef HAVE_VISIT
      std::pair< std::string, std::string >
        taskPair( batch->m_from_task->getTask()->getName(),
                  batch->m_to_tasks.front()->getTask()->getName() );

      // Check to see if this task to/from pair has been found before.
      // If not, add it to the map and insert the metrics to be
      // collected.
      if( taskPairs.find( taskPair ) == taskPairs.end() ) {

        m_comm_info[taskPair].setKeyName( "Rank" );
        m_comm_info[taskPair].insert( CommPTPMsgTo,   std::string("ToRank")  , "messages" );
        m_comm_info[taskPair].insert( CommPTPMsgFrom, std::string("FromRank"), "messages" );

        taskPairs[taskPair];
      }
#endif
      // Start the message tag with one.
      if( from == me ) {
        m_dep_batches[i]->m_message_tag = ++m_comm_info[allTasks][ to ][CommPTPMsgTo];

#ifdef HAVE_VISIT
        // Individual task comm stats.
        ++m_comm_info[taskPair][ to ][CommPTPMsgTo];
#endif  
      }
      
      if( to == me ) {
        m_dep_batches[i]->m_message_tag = ++m_comm_info[allTasks][ from ][CommPTPMsgFrom];

#ifdef HAVE_VISIT
        // Individual task comm stats.
        ++m_comm_info[taskPair][ from ][CommPTPMsgFrom];
#endif  
      }

      DOUT(g_message_tags_dbg, "Rank-" << me
           << " assigning message tag " << batch->m_message_tag
           << " from task " << batch->m_from_task->getName()
           << " to task " << batch->m_to_tasks.front()->getName()
           << ", rank-" << from << " to rank-" << to);
    }
  }

  // Loop through all of the task to/from pairs and make sure all
  // ranks are included. The ranks in the total to/from counts is the
  // union of all ranks. For visualization each individual task
  // to/from pair needs to have all ranks as well even though the
  // count is zero.
  for (auto& info: m_comm_info) {

    std::pair< std::string, std::string > taskPair = info.first;
    
    // Do the stats first so they are not affected by adding in the
    // other (possibly) unused ranks.
    info.second.calculateMinimum( true );
    info.second.reduce( false );

    if ( ((taskPair.first == "All" && taskPair.second == "Tasks") && g_message_tags_stats_dbg) ||
#ifdef HAVE_VISIT
         ((taskPair.first != "All" || taskPair.second != "Tasks") && g_message_tags_task_stats_dbg) ||
#endif
         0 ) {
      unsigned int timeStep = m_sched_common->getApplication()->getTimeStep();
      double simTime = m_sched_common->getApplication()->getSimTime();
      std::string title = "Communication TG [" + std::to_string(index) + "] " +
        "for task <" + taskPair.first + "|" + taskPair.second + ">";
      
      info.second.reportSummaryStats   ( title.c_str(), "", me,
                                         nRanks, timeStep, simTime,
                                         BaseInfoMapper::Dout, false );
      info.second.reportIndividualStats( title.c_str(), "", me,
                                         nRanks, timeStep, simTime,
                                         BaseInfoMapper::Dout );
    }

    // If the number of keys (ranks) is different then add in the
    // missing ranks (the counts will be zero).
    if( info.second.size() != m_comm_info[allTasks].size() ) {
      
      unsigned int nComms = m_comm_info[allTasks].size();
      
      for( unsigned int i=0; i<nComms; ++i) {
        
        unsigned int key = m_comm_info[allTasks].getKey(i); // rank
        
        // This call adds in the ranks and inits the value to zero.
        info.second[ key ];
      }
    }
  }
}  // end assignMessageTags()

//_____________________________________________________________________________
//
void
DetailedTasks::add( DetailedTask * dtask )
{
  m_tasks.push_back(dtask);
}

//_____________________________________________________________________________
//
void
DetailedTasks::makeDWKeyDatabase()
{
  for (auto i = 0u; i < m_local_tasks.size(); i++) {
    DetailedTask* task = m_local_tasks[i];
    //for reduction task check modifies other task check computes
    const Task::Dependency *comp = task->getTask()->isReductionTask() ? task->getTask()->getModifies() : task->getTask()->getComputes();
    for (; comp != nullptr; comp = comp->m_next) {
      const MaterialSubset* matls = comp->m_matls ? comp->m_matls : task->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable ||
            comp->m_var->typeDescription()->getType() == TypeDescription::SoleVariable) {
          m_level_keyDB.insert(comp->m_var, matl, comp->m_reduction_level);
        }
        else { // if variables saved on varDB
          const PatchSubset* patches = comp->m_patches ? comp->m_patches : task->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            m_var_keyDB.insert(comp->m_var, matl, patch);

            DOUT(g_detailed_dw_dbg, "reserve " << comp->m_var->getName() << " on Patch " << patch->getID() << ", Matl " << matl);
          }
        }
      } //end matls
    } // end comps
  } // end localtasks
}

//_____________________________________________________________________________
//
void
DetailedTasks::computeLocalTasks()
{
  int me = m_proc_group->myRank();

  if (m_local_tasks.size() != 0) {
    return;
  }

  int order = 0;
  m_initial_ready_tasks = TaskQueue();
  for (auto i = 0u; i < m_tasks.size(); i++) {
    DetailedTask* task = m_tasks[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, m_proc_group->nRanks());

    if (task->getAssignedResourceIndex() == me || task->getTask()->getType() == Task::Reduction) {
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
DetailedTasks::initializeScrubs( std::vector<OnDemandDataWarehouseP> & dws, int dwmap[] )
{
  DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Begin initialize scrubs");

  std::vector<bool> initialized(dws.size(), false);
  for (int i = 0; i < (int)Task::TotalDWs; i++) {
    if (dwmap[i] < 0) {
      continue;
    }
    OnDemandDataWarehouse* dw = dws[dwmap[i]].get_rep();
    if (dw != nullptr && dw->getScrubMode() == DataWarehouse::ScrubComplete) {
      // only a OldDW or a CoarseOldDW will have scrubComplete 
      //   But we know a future taskgraph (in a w-cycle) will need the vars if there are fine dws 
      //   between New and Old.  In this case, the scrub count needs to be complemented with CoarseOldDW
      int tgtype = getTaskGraph()->getType();
      if (!initialized[dwmap[i]] || tgtype == Scheduler::IntermediateTaskGraph) {
        // if we're intermediate, we're going to need to make sure we don't scrub CoarseOld before we finish using it
        DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Initializing scrubs on dw: " << dw->getID()
                                      << " for DW type " << i << " ADD=" << initialized[dwmap[i]]);
        dw->initializeScrubs(i, &m_scrub_count_table, initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD=" << 1);
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &m_scrub_count_table, true);
      }
      initialized[dwmap[i]] = true;
    }
  }

  DOUT(g_scrubbing_dbg, "Rank-" << Parallel::getMPIRank() << " End initialize scrubs");
}

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
  if (patch->isVirtual()) {
    patch = patch->getRealPatch();
  }
  ScrubItem key(var, matlindex, patch, dw);
  ScrubItem* result;
  result = m_scrub_count_table.lookup(&key);
  if (!result) {
    result = scinew ScrubItem(var, matlindex, patch, dw);
    m_scrub_count_table.insert(result);
  }
  result->m_count++;
  if (g_scrubbing_dbg && (var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") && (g_patch_scrub_dbg == patch->getID() || g_patch_scrub_dbg == -1)) {
    DOUT(true, "Rank-" << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/"
                       << patch->getID() << "/" << matlindex << "/" << *var << ": " << result->m_count);
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::setScrubCount( const Task::Dependency                    * req
                            ,       int                                   matl
                            , const Patch                               * patch
                            ,       std::vector<OnDemandDataWarehouseP> & dws
                            )
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

    if (g_scrubbing_dbg && (req->m_var->getName() == g_var_scrub_dbg || g_var_scrub_dbg == "") && (g_patch_scrub_dbg == patch->getID() || g_patch_scrub_dbg == -1)) {
      DOUT(true, "Rank-" << Parallel::getMPIRank() << " setting scrubcount for recv of " << req->mapDataWarehouse() << "/" << patch->getID()
                         << "/" << matl << "/" << req->m_var->getName() << ": " << scrubcount);
    }
    dws[req->mapDataWarehouse()]->setScrubCount(req->m_var, matl, patch, scrubcount);
  }
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getScrubCount( const VarLabel * label
                            ,       int        matlIndex
                            , const Patch    * patch
                            ,       int        dw
                            ,       int&       count
                            )
{
  ASSERT(!patch->isVirtual());
  ScrubItem key(label, matlIndex, patch, dw);
  ScrubItem* result = m_scrub_count_table.lookup(&key);
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
  m_scrub_count_table.remove_all();

  // Go through each of the tasks and determine which variables it will require
  for (int i = 0; i < (int)m_local_tasks.size(); i++) {
    DetailedTask* dtask = m_local_tasks[i];
    const Task* task = dtask->getTask();
    for (const Task::Dependency* req = task->getRequires(); req != nullptr; req = req->m_next) {
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
          for (unsigned int i = 0; i < neighbors.size(); i++) {
            const Patch* neighbor = neighbors[i];
            for (int m = 0; m < matls->size(); m++) {
              addScrubCount(req->m_var, matls->get(m), neighbor, whichdw);
            }
          }
        }
      }
    }

    // determine which variables this task will modify
    for (const Task::Dependency* req = task->getModifies(); req != nullptr; req = req->m_next) {
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

  if (g_scrubbing_dbg) {
    std::ostringstream message;
    message << Parallel::getMPIRank() << " scrub counts:\n";
    message << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for (FastHashTableIter<ScrubItem> iter(&m_scrub_count_table); iter.ok(); ++iter) {
      const ScrubItem* rec = iter.get_key();
      message << rec->m_dw << '/' << (rec->m_patch ? rec->m_patch->getID() : 0) << '/' << rec->m_matl << '/' << rec->m_label->getName()
               << "\t\t" << rec->m_count << '\n';
    }
    message << "Rank-" << Parallel::getMPIRank() << " end scrub counts";
    DOUT(true, message.str());
  }
}

//_____________________________________________________________________________
//
DetailedDep*
DetailedTasks::findMatchingDetailedDep(       DependencyBatch  * batch
                                      ,       DetailedTask     * toTask
                                      ,       Task::Dependency * req
                                      , const Patch            * fromPatch
                                      ,       int                matl
                                      ,       IntVector          low
                                      ,       IntVector          high
                                      ,       IntVector        & totalLow
                                      ,       IntVector        & totalHigh
                                      ,       DetailedDep     *& parent_dep
                                      )
{
  totalLow = low;
  totalHigh = high;
  DetailedDep* dep = batch->m_head;

  parent_dep = nullptr;
  DetailedDep* last_dep = nullptr;
  DetailedDep* valid_dep = nullptr;

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
DetailedTasks::possiblyCreateDependency(       DetailedTask     * from
                                       ,       Task::Dependency * comp
                                       , const Patch            * fromPatch
                                       ,       DetailedTask     * to
                                       ,       Task::Dependency * req
                                       , const Patch            * toPatch
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

  if ((toresource == my_rank || (req->m_patches_dom != Task::ThisLevel && fromresource == my_rank))
      && fromPatch && !req->m_var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->m_var, matl, fromPatch, req->m_whichdw);
  }

  // if the dependency is on the same processor then add an internal dependency
  if (fromresource == my_rank && fromresource == toresource) {
    to->addInternalDependency(from, req->m_var);
    return;
  }

  // this should have been pruned out earlier
  ASSERT(!req->m_var->typeDescription()->isReductionVariable())
  // Do not check external deps on SoleVariable
  if (req->m_var->typeDescription()->getType() == TypeDescription::SoleVariable) {
    return;
  }

  // make keys for MPI messages
  if (fromPatch) {
    m_var_keyDB.insert(req->m_var, matl, fromPatch);
  }

  // get dependency batch
  DependencyBatch* batch = from->getComputes();

  // find dependency batch that is to the same processor as this dependency
  for (; batch != nullptr; batch = batch->m_comp_next) {
    if (batch->m_to_rank == toresource) {
      break;
    }
  }

  // if batch doesn't exist then create it
  if (!batch) {
    batch = scinew DependencyBatch(toresource, from, to);
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
  DetailedDep* new_dep = scinew DetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  // search for a dependency that can be combined with this dependency

  // location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl,
                                                      new_dep->m_low, new_dep->m_high,
                                                      varRangeLow, varRangeHigh, parent_dep);

  // This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDep* insert_dep = parent_dep;

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
    if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x") {
        DOUT(g_detailed_tasks_dbg, "Rank-" << my_rank << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
                                           << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
                                           << " cond " << cond << " dw " << req->mapDataWarehouse());
      }

      if (fromresource == my_rank) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter != m_particle_sends[toresource].end());

        //subtract one from the count
        iter->count_--;

        // if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
        }
      }
      else if (toresource == my_rank) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
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
  if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == my_rank) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
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
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
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
DetailedTask*
DetailedTasks::getOldDWSendTask( int proc )
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
DetailedTasks::internalDependenciesSatisfied( DetailedTask * dtask )
{
  std::lock_guard<Uintah::MasterLock> internal_deps_satisfied_guard(g_internal_ready_mutex);

  m_ready_tasks.push(dtask);
  m_atomic_initial_ready_tasks_size.fetch_add(1, std::memory_order_relaxed);
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  std::lock_guard<Uintah::MasterLock> internal_ready_guard(g_internal_ready_mutex);

  DetailedTask* nextTask = nullptr;
  if (m_atomic_initial_ready_tasks_size.load(std::memory_order_acquire) > 0) {
    if (!m_ready_tasks.empty()) {
      nextTask = m_ready_tasks.front();
      m_atomic_initial_ready_tasks_size.fetch_sub(1, std::memory_order_relaxed);
      m_ready_tasks.pop();
    }
  }

  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numInternalReadyTasks()
{
  return m_atomic_initial_ready_tasks_size.load(std::memory_order_seq_cst);
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  std::lock_guard<Uintah::MasterLock> external_ready_guard(g_external_ready_mutex);

  DetailedTask* nextTask = nullptr;
  if (m_atomic_mpi_completed_tasks_size.load(std::memory_order_acquire) > 0) {
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
DetailedTasks::numExternalReadyTasks()
{
  return m_atomic_mpi_completed_tasks_size.load(std::memory_order_seq_cst);
}

//_____________________________________________________________________________
//
void
DetailedTasks::initTimestep()
{
  m_ready_tasks = m_initial_ready_tasks;
  m_atomic_initial_ready_tasks_size.store(m_initial_ready_tasks.size(), std::memory_order_release);
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
  for (int i = 0; i < static_cast<int>(m_dep_batches.size()); i++) {
    m_dep_batches[i]->reset();
  }
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
  logMemory(out, total, tag, "tasks", "DetailedTask", nullptr, -1, elems1.str(), m_tasks.size() * sizeof(DetailedTask), 0);
  std::ostringstream elems2;
  elems2 << m_dep_batches.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", nullptr, -1, elems2.str(), m_dep_batches.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)m_dep_batches.size(); i++) {
    for (DetailedDep* dep = m_dep_batches[i]->m_head; dep != nullptr; dep = dep->m_next) {
      ndeps++;
    }
  }
  std::ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "DetailedDep", nullptr, -1, elems3.str(), ndeps * sizeof(DetailedDep), 0);
}

//_____________________________________________________________________________
//
void
DetailedTasks::emitEdges( ProblemSpecP edgesElement, int rank )
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
DetailedTaskPriorityComparison::operator()( DetailedTask *& ltask
                                          , DetailedTask *& rtask
                                          )
{
  QueueAlg alg = ltask->getTaskGroup()->getTaskPriorityAlg();
  ASSERT(alg == rtask->getTaskGroup()->getTaskPriorityAlg());

  if (alg == FCFS) {
    return false;               // First Come First Serve;
  }

  else if (alg == Stack) {
    return true;               // First Come Last Serve, for robust testing;
  }

  else if (alg == Random) {
    return (random() % 2 == 0);   // Random;
  }

  else if (alg == MostChildren) {
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
    std::set<Task*>::iterator it;
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
    for (DependencyBatch* batch = ltask->getComputes(); batch != nullptr; batch = batch->m_comp_next) {
      for (DetailedDep* dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
        lmsg++;
      }
    }
    for (DependencyBatch* batch = rtask->getComputes(); batch != nullptr; batch = batch->m_comp_next) {
      for (DetailedDep* dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
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
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    // send_old_data task will have a nullptr PatchSubset, there will be only one per node
    if (lpatches == nullptr) {
      return true;
    }
    if (rpatches == nullptr) {
      return false;
    }
    // otherwise do the standard comparison
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
    const PatchSubset* lpatches = ltask->getPatches();
    const PatchSubset* rpatches = rtask->getPatches();
    // send_old_data task will have a nullptr PatchSubset, there will be only one per node
    if (lpatches == nullptr) {
      return true;
    }
    if (rpatches == nullptr) {
      return false;
    }
    // otherwise do the standard comparison
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

  // Will later reincorporate this correctly, AscendingStaticOrder and DecendingStaticOrder
//  if (ltask->getTask()->getSortedOrder() > rtask->getTask()->getSortedOrder()) {
//    return true;
//  }
//
//  if (ltask->getTask()->getSortedOrder() < rtask->getTask()->getSortedOrder()) {
//    return false;
//  }

}



#ifdef HAVE_CUDA

//_____________________________________________________________________________
//
bool
DetailedTasks::getDeviceValidateRequiresAndModifiesCopiesTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_validateRequiresAndModifiesCopies_pool_iter = device_validateRequiresAndModifiesCopies_pool.find_any(ready_request);

  if (device_validateRequiresAndModifiesCopies_pool_iter) {
    dtask = *device_validateRequiresAndModifiesCopies_pool_iter;
    device_validateRequiresAndModifiesCopies_pool.erase(device_validateRequiresAndModifiesCopies_pool_iter);
    //printf("device_validateRequiresAndModifiesCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), device_validateRequiresAndModifiesCopies_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getDevicePerformGhostCopiesTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
DetailedTasks::getDeviceValidateGhostCopiesTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
DetailedTasks::getDeviceCheckIfExecutableTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
DetailedTasks::getDeviceReadyToExecuteTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator device_readyToExecute_pool_iter = device_readyToExecute_pool.find_any(ready_request);
  if (device_readyToExecute_pool_iter) {
    dtask = *device_readyToExecute_pool_iter;
    int task_to_debug_threshold = Uintah::Parallel::getAmountTaskNameExpectedToRun();

    bool proceed{true};
    if (task_to_debug_threshold > 0) {
      std::string task_to_debug_name = Uintah::Parallel::getTaskNameToTime();
      std::string current_task = dtask->getTask()->getName();
      int task_to_debug_count = atomic_task_to_debug_size.load(std::memory_order_relaxed);
      if ( current_task.size() >= task_to_debug_name.size() 
           && dtask->getTask()->getName().substr(0, task_to_debug_name.size()) == task_to_debug_name ) {
        if ( task_to_debug_count % task_to_debug_threshold != 0 ) {
          proceed = false;
        } 
      }
    }
    if (proceed) {
      device_readyToExecute_pool.erase(device_readyToExecute_pool_iter);
      retVal = true;
    }
  }
  return retVal;
}


//______________________________________________________________________
//
bool
DetailedTasks::getDeviceExecutionPendingTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
DetailedTasks::getHostValidateRequiresAndModifiesCopiesTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
  TaskPool::iterator host_validateRequiresAndModifiesCopies_pool_iter = host_validateRequiresAndModifiesCopies_pool.find_any(ready_request);

  if (host_validateRequiresAndModifiesCopies_pool_iter) {
    dtask = *host_validateRequiresAndModifiesCopies_pool_iter;
    host_validateRequiresAndModifiesCopies_pool.erase(host_validateRequiresAndModifiesCopies_pool_iter);
    //printf("host_validateRequiresAndModifiesCopies_pool - Erased %s size of pool %lu\n", dtask->getName().c_str(), host_validateRequiresAndModifiesCopies_pool.size());
    retVal = true;
  }

  return retVal;
}

//_____________________________________________________________________________
//
bool
DetailedTasks::getHostCheckIfExecutableTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
DetailedTasks::getHostReadyToExecuteTask(DetailedTask *& dtask)
{
  //This function should ONLY be called within runTasks() part 1.
  //This is all done as one atomic unit as we're seeing if we should get an item and then we get it.
  bool retVal = false;
  dtask = nullptr;

  auto ready_request = [](DetailedTask *& dtask)->bool { return dtask->checkAllCudaStreamsDoneForThisTask(); };
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
void DetailedTasks::addDeviceValidateRequiresAndModifiesCopies(DetailedTask * dtask)
{
  device_validateRequiresAndModifiesCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::addDevicePerformGhostCopies(DetailedTask * dtask)
{
  device_performGhostCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::addDeviceValidateGhostCopies(DetailedTask * dtask)
{
  device_validateGhostCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::addDeviceCheckIfExecutable(DetailedTask * dtask)
{
  device_checkIfExecutable_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
DetailedTasks::addDeviceReadyToExecute( DetailedTask * dtask )
{
  device_readyToExecute_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
DetailedTasks::addDeviceExecutionPending( DetailedTask * dtask )
{
  device_executionPending_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::addHostValidateRequiresAndModifiesCopies(DetailedTask * dtask)
{
  host_validateRequiresAndModifiesCopies_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::addHostCheckIfExecutable(DetailedTask * dtask)
{
  host_checkIfExecutable_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void
DetailedTasks::addHostReadyToExecute( DetailedTask * dtask )
{
  host_readyToExecute_pool.insert(dtask);
}

//_____________________________________________________________________________
//
void DetailedTasks::createInternalDependencyBatch(       DetailedTask     * from
                                                 ,       Task::Dependency * comp
                                                 , const Patch            * fromPatch
                                                 ,       DetailedTask     * to
                                                 ,       Task::Dependency * req
                                                 , const Patch            * toPatch
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
  DetailedDep* new_dep = scinew DetailedDep(batch->m_head, comp, req, to, fromPatch, matl, low, high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->m_low, new_dep->m_high,
                                                              varRangeLow, varRangeHigh, parent_dep);

  //This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDep* insert_dep = parent_dep;
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
    if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->m_low, matching_dep->m_high, (int)cond);

      if (req->m_var->getName() == "p.x")
        DOUT(g_detailed_tasks_dbg, "Rank-" << m_proc_group->myRank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->m_var
                          << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->m_low << " " << matching_dep->m_high
                          << " cond " << cond << " dw " << req->mapDataWarehouse());

      if (fromresource == m_proc_group->myRank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
        ASSERT(iter!=m_particle_sends[toresource].end());

        //subtract one from the count
        iter->count_--;

        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          m_particle_sends[toresource].erase(iter);
        }
      } else if (toresource == m_proc_group->myRank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
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
  if (req->m_var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->m_whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->m_low, new_dep->m_high, (int)cond, 1);

    if (fromresource == m_proc_group->myRank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_sends[toresource].find(pmg);
      if (iter == m_particle_sends[toresource].end())  //if does not exist
          {
        //add to the sends list
        m_particle_sends[toresource].insert(pmg);
      } else {
        //increment count
        iter->count_++;
      }
    } else if (toresource == m_proc_group->myRank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = m_particle_recvs[fromresource].find(pmg);
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
DetailedDep* DetailedTasks::findMatchingInternalDetailedDep(DependencyBatch         * batch
                                                           ,       DetailedTask     * toTask
                                                           ,       Task::Dependency * req
                                                           , const Patch            * fromPatch
                                                           ,       int                matl
                                                           ,       IntVector          low
                                                           ,       IntVector          high
                                                           ,       IntVector        & totalLow
                                                           ,       IntVector        & totalHigh
                                                           ,       DetailedDep     *& parent_dep
                                                           )
{
  totalLow   = low;
  totalHigh  = high;
  parent_dep = nullptr;

  DetailedDep * dep       = batch->m_head;
  DetailedDep * last_dep  = nullptr;
  DetailedDep * valid_dep = nullptr;

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
