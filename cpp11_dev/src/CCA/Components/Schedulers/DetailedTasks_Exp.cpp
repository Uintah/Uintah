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

#include <CCA/Components/Schedulers/DetailedTasks_Exp.hpp>

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>

// sync cout/cerr so they are readable when output by multiple threads
#include <sci_defs/config_defs.h>
#include <sci_defs/cuda_defs.h>


namespace Uintah {


namespace {

Dout mpidbg{         "MPIDBG"          , false };
Dout dbg{            "DetailedTasks"   , false };
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
  sc_(sc),
  d_myworld(pg),
  first(first),
  taskgraph_(taskgraph),
  mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
  currentDependencyGeneration_(1),
  extraCommunication_(0)
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

  stask_ = scinew Task( "send old data", Task::InitialSend );
  stask_->d_phase = 0;
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
    int from = batch->fromTask->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, d_myworld->size());
    int to = batch->to;
    ASSERTRANGE(to, 0, d_myworld->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      std::pair<int, int> fromToPair = std::make_pair(from, to);
      batches_[i]->messageTag = ++perPairBatchIndices[fromToPair];  // start with one
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
    DetailedTask* dtask = localtasks_[i];
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
          levelKeyDB.insert(comp->var, matl, comp->reductionLevel);
        }
        else { // if variables saved on varDB
          const PatchSubset* patches = comp->patches ? comp->patches : dtask->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            varKeyDB.insert(comp->var, matl, patch);
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
void
DetailedTasks::initializeScrubs( std::vector<OnDemandDataWarehouseP>& dws,
                                 int                             dwmap[] )
{
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
        dw->initializeScrubs(i, &(first ? first->scrubCountTable_ : scrubCountTable_), initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &(first ? first->scrubCountTable_ : scrubCountTable_), true);
      }
      initialized[dwmap[i]] = true;
    }
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
  result = (first ? first->scrubCountTable_ : scrubCountTable_).lookup(&key);
  if (!result) {
    result = scinew ScrubItem(var, matlindex, patch, dw);
    (first ? first->scrubCountTable_ : scrubCountTable_).insert(result);
  }
  result->count++;
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
  ScrubItem* result = (first ? first->scrubCountTable_ : scrubCountTable_).lookup(&key);
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
  (first ? first->scrubCountTable_ : scrubCountTable_).remove_all();

  // Go through each of the tasks and determine which variables it will require
  for (int i = 0; i < (int)localtasks_.size(); i++) {
    DetailedTask* dtask = localtasks_[i];
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
}

//_____________________________________________________________________________
//

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
  DetailedDep* dep = batch->head;

  parent_dep = 0;
  DetailedDep* last_dep = 0;
  DetailedDep* valid_dep = 0;

  //search each dep
  for (; dep != 0; dep = dep->next) {
    //if deps are equivalent
    if (fromPatch == dep->fromPatch && matl == dep->matl
        && (req == dep->req || (req->var->equals(dep->req->var) && req->mapDataWarehouse() == dep->req->mapDataWarehouse()))) {

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
          if (dep->low == new_l && dep->high == new_h) {
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

  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();

  // if neither task talks to this processor, return
  if (fromresource != d_myworld->myrank() && toresource != d_myworld->myrank()) {
    return;
  }

  if ((toresource == d_myworld->myrank() || (req->patches_dom != Task::ThisLevel && fromresource == d_myworld->myrank()))
      && fromPatch && !req->var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->var, matl, fromPatch, req->whichdw);
  }

  //if the dependency is on the same processor then add an internal dependency
  if (fromresource == d_myworld->myrank() && fromresource == toresource) {
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
  if (fromPatch) varKeyDB.insert(req->var,matl,fromPatch);

  //get dependency batch
  DependencyBatch* batch = from->getComputes();

  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->comp_next) {
    if (batch->to == toresource) {
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
  }
  else if (mustConsiderInternalDependencies_) {  // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add to the batch's toTasks.
      batch->toTasks.push_back(to);
    }
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  // create the new dependency
  DetailedDep* new_dep = scinew DetailedDep(batch->head, comp, req, to, fromPatch, matl, low, high, cond);

  // search for a dependency that can be combined with this dependency

  // location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->low, new_dep->high, varRangeLow,
                                                      varRangeHigh, parent_dep);

  // This is set to either the parent of the first matching dep or when there is no matching deps the last dep in the list.
  DetailedDep* insert_dep = parent_dep;

  // if we have matching dependencies we will extend the new dependency to include the old one and delete the old one
  while (matching_dep != 0) {

    // extend the dependency range
    new_dep->low = Min(new_dep->low, matching_dep->low);
    new_dep->high = Max(new_dep->high, matching_dep->high);

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
    new_dep->toTasks.splice(new_dep->toTasks.begin(), matching_dep->toTasks);

    // erase particle sends/recvs
    if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->low, matching_dep->high, (int)cond);

      if (fromresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
        ASSERT(iter != particleSends_[toresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          particleSends_[toresource].erase(iter);
        }
      }
      else if (toresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
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
    if (parent_dep == NULL) {
      batch->head = matching_dep->next;
    }
    else {
      parent_dep->next = matching_dep->next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, new_dep->low, new_dep->high,
                                           varRangeLow, varRangeHigh, parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep) {
      insert_dep = parent_dep;
    }
  }

  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->patchLow = varRangeLow;
  new_dep->patchHigh = varRangeHigh;

  if (insert_dep == NULL) {
    //no dependencies are in the list so add it to the head
    batch->head = new_dep;
    new_dep->next = NULL;
  }
  else {
    //dependencies already exist so add it at the insert location.
    new_dep->next = insert_dep->next;
    insert_dep->next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->low, new_dep->high, (int)cond, 1);

    if (fromresource == d_myworld->myrank()) {
      std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
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
      std::set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
      if (iter == particleRecvs_[fromresource].end()) {
        //add to the recvs list
        particleRecvs_[fromresource].insert(pmg);
      }
      else {
        //increment the count
        iter->count_++;
      }

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
  //get dependancy batch
  DependencyBatch* batch = from->getInternalComputes();
  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();
  //find dependency batch that is to the same processor as this dependency
  for (; batch != 0; batch = batch->comp_next) {
    if (batch->to == toresource)
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
      batch->toTasks.push_back(to);
    }
    if (dbg.active())
      dbg << d_myworld->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
  }

  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);

  //create the new dependency
  DetailedDep* new_dep = scinew DetailedDep(batch->head, comp, req, to, fromPatch, matl, low, high, cond);

  //search for a dependency that can be combined with this dependency

  //location of parent dependency
  DetailedDep* parent_dep;

  DetailedDep* matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->low, new_dep->high, varRangeLow,
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
      dbg << d_myworld->myrank() << "            EXTENDED from " << new_dep->low << " " << new_dep->high << " to "
          << Min(new_dep->low, matching_dep->low) << " " << Max(new_dep->high, matching_dep->high) << "\n";
      dbg << *req->var << '\n';
      dbg << *new_dep->req->var << '\n';
      if (comp)
        dbg << *comp->var << '\n';
      if (new_dep->comp)
        dbg << *new_dep->comp->var << '\n';
    }



    //extend the dependency range
    new_dep->low = Min(new_dep->low, matching_dep->low);
    new_dep->high = Max(new_dep->high, matching_dep->high);


    //copy matching dependencies toTasks to the new dependency
    new_dep->toTasks.splice(new_dep->toTasks.begin(), matching_dep->toTasks);

    //erase particle sends/recvs
    if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
      PSPatchMatlGhostRange pmg(fromPatch, matl, matching_dep->low, matching_dep->high, (int)cond);

      if (req->var->getName() == "p.x")
        dbg << d_myworld->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->var
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->low << " " << matching_dep->high
            << " cond " << cond << " dw " << req->mapDataWarehouse() << std::endl;

      if (fromresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
        ASSERT(iter!=particleSends_[toresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          particleSends_[toresource].erase(iter);
          //particleSends_[toresource].erase(pmg);
        }
      } else if (toresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleRecvs_[fromresource].find(pmg);
        ASSERT(iter!=particleRecvs_[fromresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the recvs list
        if (iter->count_ == 0) {
          particleRecvs_[fromresource].erase(iter);
          //particleRecvs_[fromresource].erase(pmg);
        }
      }
    }

    //remove the matching_dep from the batch list
    if (parent_dep == NULL) {
      batch->head = matching_dep->next;
    } else {
      parent_dep->next = matching_dep->next;
    }

    //delete matching dep
    delete matching_dep;

    //search for another matching detailed deps
    matching_dep = findMatchingInternalDetailedDep(batch, to, req, fromPatch, matl, new_dep->low, new_dep->high, varRangeLow, varRangeHigh,
                                           parent_dep);

    //if the matching dep is the current insert dep then we must move the insert dep to the new parent dep
    if (matching_dep == insert_dep)
      insert_dep = parent_dep;
  }



  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  new_dep->patchLow = varRangeLow;
  new_dep->patchHigh = varRangeHigh;


  if (insert_dep == NULL) {
    //no dependencies are in the list so add it to the head
    batch->head = new_dep;
    new_dep->next = NULL;
  } else {
    //depedencies already exist so add it at the insert location.
    new_dep->next = insert_dep->next;
    insert_dep->next = new_dep;
  }

  //add communication for particle data
  // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
  if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
    PSPatchMatlGhostRange pmg = PSPatchMatlGhostRange(fromPatch, matl, new_dep->low, new_dep->high, (int)cond, 1);

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
    if (req->var->getName() == "p.x")
      dbg << d_myworld->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
          << req->mapDataWarehouse() << std::endl;
  }

  if (dbg.active()) {
    dbg << d_myworld->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch)
      dbg << fromPatch->getID() << '\n';
    else
      dbg << "NULL\n";
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
  DetailedDep* dep = batch->head;

  parent_dep = 0;
  DetailedDep* last_dep = 0;
  DetailedDep* valid_dep = 0;
  //For now, turning off a feature that can combine ghost cells into larger vars
  //for scenarios where one source ghost cell var can handle more than one destination patch.

  //search each dep
  for (; dep != 0; dep = dep->next) {
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
DetailedTasks::internalDependenciesSatisfied( DetailedTask* task )
{
  readyQueueLock_.lock();
  {
    readyTasks_.push(task);
  }
  readyQueueLock_.unlock();
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  readyQueueLock_.lock();
  {
    if (!readyTasks_.empty()) {
      nextTask = readyTasks_.front();
      readyTasks_.pop();
    }
  }
  readyQueueLock_.unlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numInternalReadyTasks()
{
  int size = 0;
  readyQueueLock_.lock();
  {
    size = readyTasks_.size();
  }
  readyQueueLock_.unlock();
  return size;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  mpiCompletedQueueLock_.lock();
  {
    if (!mpiCompletedTasks_.empty()) {
      nextTask = mpiCompletedTasks_.top();
      mpiCompletedTasks_.pop();
    }
  }
  mpiCompletedQueueLock_.unlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numExternalReadyTasks()
{
  int size = 0;
  mpiCompletedQueueLock_.lock();
  {
    size = mpiCompletedTasks_.size();
  }
  mpiCompletedQueueLock_.unlock();
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
DetailedTasks::logMemoryUse(       std::ostream&       out,
                                   unsigned long& total,
                             const std::string&        tag )
{
  std::ostringstream elems1;
  elems1 << tasks_.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1, elems1.str(), tasks_.size() * sizeof(DetailedTask), 0);
  std::ostringstream elems2;
  elems2 << batches_.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1, elems2.str(), batches_.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)batches_.size(); i++) {
    for (DetailedDep* p = batches_[i]->head; p != 0; p = p->next) {
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
    DetailedTask* fromTask = (*req_iter).first->fromTask;
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
    for (DependencyBatch* batch = ltask->getComputes(); batch != 0; batch = batch->comp_next) {
      for (DetailedDep* dep = batch->head; dep != 0; dep = dep->next) {
        lmsg++;
      }
    }
    for (DependencyBatch* batch = rtask->getComputes(); batch != 0; batch = batch->comp_next) {
      for (DetailedDep* dep = batch->head; dep != 0; dep = dep->next) {
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
