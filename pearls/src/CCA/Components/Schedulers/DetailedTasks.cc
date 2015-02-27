/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/CommRecMPI.h>
#include <CCA/Components/Schedulers/MemoryLog.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>
#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

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
static string dbgScrubVar   = "";
static int    dbgScrubPatch = -1;

std::map<string, double> DependencyBatch::waittimes;

//_____________________________________________________________________________
//
DetailedTasks::DetailedTasks(       SchedulerCommon* sc,
                              const ProcessorGroup*  pg,
                                    DetailedTasks*   first,
                              const TaskGraph*       taskgraph,
                              const std::set<int>&   neighborhood_processors,
                                    bool             mustConsiderInternalDependencies /*= false*/ )
  :   sc_(sc),
      d_myworld(pg),
      first(first),
      taskgraph_(taskgraph),
      mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
      currentDependencyGeneration_(1),
      extraCommunication_(0),
      readyQueueLock_("DetailedTasks Ready Queue"),
      mpiCompletedQueueLock_("DetailedTasks MPI completed Queue")
#ifdef HAVE_CUDA
      ,deviceReadyQueueLock_("DetailedTasks Device Ready Queue"),
      deviceCompletedQueueLock_("DetailedTasks Device Completed Queue")
#endif
{
  // Set up mappings for the initial send tasks
  int dwmap[Task::TotalDWs];
  for (int i = 0; i < Task::TotalDWs; i++) {
    dwmap[i] = Task::InvalidDW;
  }
  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = Task::NoDW;

  stask_ = scinew Task("send old data", Task::InitialSend);
  stask_->d_phase = 0;
  stask_->setMapping(dwmap);

  //create a send old detailed task for every processor in my neighborhood
  for (set<int>::iterator iter = neighborhood_processors.begin(); iter != neighborhood_processors.end(); iter++) {
    DetailedTask* newtask = scinew DetailedTask(stask_, 0, 0, this);
    newtask->assignResource(*iter);
    //use a map because the processors in this map are likely to be sparse
    sendoldmap[*iter] = tasks_.size();
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
DependencyBatch::~DependencyBatch()
{
  DetailedDep* dep = head;
  while (dep) {
    DetailedDep* tmp = dep->next;
    delete dep;
    dep = tmp;
  }
  delete lock_;
}

//_____________________________________________________________________________
//
void
DetailedTasks::assignMessageTags( int me )
{
  // maps from, to (process) pairs to indices for each batch of that pair
  map<pair<int, int>, int> perPairBatchIndices;

  for (int i = 0; i < (int)batches_.size(); i++) {
    DependencyBatch* batch = batches_[i];
    int from = batch->fromTask->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, d_myworld->size());
    int to = batch->to;
    ASSERTRANGE(to, 0, d_myworld->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing perPairBatchIndices.
      pair<int, int> fromToPair = make_pair(from, to);
      batches_[i]->messageTag = ++perPairBatchIndices[fromToPair];  // start with one
      if (messagedbg.active()) {
        coutLock.lock();
        messagedbg << me << " assigning message num " << batch->messageTag << " from task " << batch->fromTask->getName()
                   << " to task " << batch->toTasks.front()->getName() << ", process " << from << " to process " << to << "\n";
        coutLock.unlock();
      }
    }
  }

  if (dbg.active()) {
    map<pair<int, int>, int>::iterator iter;
    for (iter = perPairBatchIndices.begin(); iter != perPairBatchIndices.end(); iter++) {
      int from = iter->first.first;
      int to = iter->first.second;
      int num = iter->second;
      dbg << num << " messages from process " << from << " to process " << to << "\n";
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
    for (; comp != 0; comp = comp->next) {
      const MaterialSubset* matls = comp->matls ? comp->matls : task->getMaterials();
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        // if variables saved on levelDB
        if (comp->var->typeDescription()->getType() == TypeDescription::ReductionVariable ||
            comp->var->typeDescription()->getType() == TypeDescription::SoleVariable) {
          levelKeyDB.insert(comp->var, matl, comp->reductionLevel);
        }
        else { // if variables saved on varDB
          const PatchSubset* patches = comp->patches ? comp->patches : task->getPatches();
          for (int p = 0; p < patches->size(); p++) {
            const Patch* patch = patches->get(p);
            varKeyDB.insert(comp->var, matl, patch);
            if (dwdbg.active()) {
              dwdbg << "reserve " << comp->var->getName() << " on Patch " << patch->getID() << ", Matl " << matl << std::endl;
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
DetailedTask::DetailedTask(       Task*           task,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DetailedTasks*  taskGroup )
  :   task(task),
      patches(patches),
      matls(matls),
      comp_head(0),
      taskGroup(taskGroup),
      numPendingInternalDependencies(0),
      internalDependencyLock("DetailedTask Internal Dependencies"),
      resourceIndex(-1),
      staticOrder(-1),
      d_profileType(Normal)
{
  if (patches) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(patches->getVector().begin(), patches->getVector().end(), Patch::Compare()));
    patches->addReference();
  }
  if (matls) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(matls->getVector().begin(), matls->getVector().end()));
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
DetailedTask::doit( const ProcessorGroup*                      pg,
                          std::vector<OnDemandDataWarehouseP>& oddws,
                          std::vector<DataWarehouseP>&         dws,
                          Task::CallBackEvent                  event /* = Task::CPU */ )
{
  TAU_PROFILE("DetailedTask::doit", " ", TAU_USER);
  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "DetailedTask " << this << " begin doit()\n";
    mixedDebug << " task is " << task << "\n";
    mixedDebug << "   num Pending Deps: " << numPendingInternalDependencies << "\n";
    mixedDebug << "   Originally needed deps (" << internalDependencies.size() << "):\n";

    list<InternalDependency>::iterator iter = internalDependencies.begin();

    for (int i = 0; iter != internalDependencies.end(); iter++, i++) {
      mixedDebug << i << ":    " << *((*iter).prerequisiteTask->getTask()) << "\n";
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
    cudaError_t retVal;
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(deviceNum_));
    task->doit(event, pg, patches, matls, dws, d_cudaStream, deviceNum_);
  }
  else {
    task->doit(event, pg, patches, matls, dws, NULL, -1);
  }
#else
  task->doit(event, pg, patches, matls, dws, NULL, -1);
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
DetailedTasks::initializeScrubs( std::vector<OnDemandDataWarehouseP>& dws,
                                 int                                  dwmap[] )
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
DetailedTask::scrub( vector<OnDemandDataWarehouseP>& dws )
{
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " Starting scrub after task: " << *this << '\n';
  }
  const Task* task = getTask();

  const set<const VarLabel*, VarLabel::Compare>& initialRequires = taskGroup->getSchedulerCommon()->getInitialRequiredVars();
  const set<string>& unscrubbables = taskGroup->getSchedulerCommon()->getNoScrubVars();

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
                cout << "   BAD BOY FROM Task : " << *this << " scrubbing " << *req << " PATCHES: " << *patches.get_rep() << endl;
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
            if (taskGroup->getScrubCount(comp->var, matl, patch, whichdw, count)) {
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
  if (scrubout.active() && (var->getName() == dbgScrubVar || dbgScrubVar == "")
      && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1)) {
    scrubout << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/" << patch->getID() << "/" << matlindex
             << "/" << *var << ": " << result->count << endl;
  }
}

//_____________________________________________________________________________
//
void
DetailedTasks::setScrubCount( const Task::Dependency*                    req,
                                    int                                  matl,
                              const Patch*                               patch,
                                    std::vector<OnDemandDataWarehouseP>& dws )
{
  ASSERT(!patch->isVirtual());
  DataWarehouse::ScrubMode scrubmode = dws[req->mapDataWarehouse()]->getScrubMode();
  const set<const VarLabel*, VarLabel::Compare>& initialRequires = getSchedulerCommon()->getInitialRequiredVars();
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
  if (scrubout.active()) {
    scrubout << Parallel::getMPIRank() << " scrub counts:\n";
    scrubout << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for (FastHashTableIter<ScrubItem> iter(&(first ? first->scrubCountTable_ : scrubCountTable_)); iter.ok(); ++iter) {
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
DetailedTask::findRequiringTasks( const VarLabel*                 var,
                                        std::list<DetailedTask*>& requiringTasks )
{
  // find requiring tasks

  // find external requires
  for (DependencyBatch* batch = getComputes(); batch != 0; batch = batch->comp_next) {
    for (DetailedDep* dep = batch->head; dep != 0; dep = dep->next) {
      if (dep->req->var == var) {
        requiringTasks.insert(requiringTasks.end(), dep->toTasks.begin(), dep->toTasks.end());
      }
    }
  }

  // find internal requires
  map<DetailedTask*, InternalDependency*>::iterator internalDepIter;
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
        else if (dbg.active()) {
          dbg << d_myworld->myrank() << "            Ignoring: " << dep->low << " " << dep->high << ", fromPatch = ";
          if (fromPatch) {
            dbg << fromPatch->getID() << '\n';
          }
          else {
            dbg << "NULL\n";
          }
          dbg << d_myworld->myrank() << " TP: " << totalLow << " " << totalHigh << endl;
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
  TAU_PROFILE("DetailedTasks::possiblyCreateDependency", " ", TAU_USER);

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

  if ((toresource == d_myworld->myrank() || (req->patches_dom != Task::ThisLevel && fromresource == d_myworld->myrank()))
      && fromPatch && !req->var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->var, matl, fromPatch, req->whichdw);
  }

  //if the dependency is on the same processor then add an internal dependency
  if (fromresource == d_myworld->myrank() && fromresource == toresource) {
    to->addInternalDependency(from, req->var);
    return;
  }

  //this should have been pruned out earlier
  ASSERT(!req->var->typeDescription()->isReductionVariable())
  // Do not check external deps on SoleVariable
  if (req->var->typeDescription()->getType() == TypeDescription::SoleVariable) {
    return;
  }

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
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "          NEW BATCH!\n";
    }
  }
  else if (mustConsiderInternalDependencies_) {  // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add to the batch's toTasks.
      batch->toTasks.push_back(to);
    }
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
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

    //debugging output
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "            EXTENDED from " << new_dep->low << " " << new_dep->high << " to "
          << Min(new_dep->low, matching_dep->low) << " " << Max(new_dep->high, matching_dep->high) << "\n";
      dbg << *req->var << '\n';
      dbg << *new_dep->req->var << '\n';
      if (comp) {
        dbg << *comp->var << '\n';
      }
      if (new_dep->comp) {
        dbg << *new_dep->comp->var << '\n';
      }
    }

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

      if (req->var->getName() == "p.x") {
        dbg << d_myworld->myrank() << " erasing particles from " << fromresource << " to " << toresource << " var " << *req->var
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << matching_dep->low << " " << matching_dep->high
            << " cond " << cond << " dw " << req->mapDataWarehouse() << std::endl;
      }

      if (fromresource == d_myworld->myrank()) {
        std::set<PSPatchMatlGhostRange>::iterator iter = particleSends_[toresource].find(pmg);
        ASSERT(iter != particleSends_[toresource].end());
        //subtract one from the count
        iter->count_--;
        //if the count is zero erase it from the sends list
        if (iter->count_ == 0) {
          particleSends_[toresource].erase(iter);
//          particleSends_[toresource].erase(pmg);
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
//          particleRecvs_[fromresource].erase(pmg);
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
    if (req->var->getName() == "p.x") {
      dbg << d_myworld->myrank() << " scheduling particles from " << fromresource << " to " << toresource << " on patch "
          << fromPatch->getID() << " matl " << matl << " range " << low << " " << high << " cond " << cond << " dw "
          << req->mapDataWarehouse() << std::endl;
    }
  }

  if (dbg.active()) {
    dbg << d_myworld->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
    if (fromPatch) {
      dbg << fromPatch->getID() << '\n';
    }
    else {
      dbg << "NULL\n";
    }
  }
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getOldDWSendTask( int proc )
{
#if SCI_ASSERTION_LEVEL>0
  //verify the map entry has been created
  if (sendoldmap.find(proc) == sendoldmap.end()) {
    cout << d_myworld->myrank() << " Error trying to get oldDWSendTask for processor: " << proc << " but it does not exist\n";
    throw InternalError("oldDWSendTask does not exist", __FILE__, __LINE__);
  }
#endif 
  return tasks_[sendoldmap[proc]];
}

//_____________________________________________________________________________
//
void
DetailedTask::addComputes( DependencyBatch* comp )
{
  comp->comp_next = comp_head;
  comp_head = comp;
}

//_____________________________________________________________________________
//
bool
DetailedTask::addRequires( DependencyBatch* req )
{
  // return true if it is adding a new batch
  return reqs.insert(make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
// can be called in one of two places - when the last MPI Recv has completed, or from MPIScheduler
void
DetailedTask::checkExternalDepCount()
{
  if (mpidbg.active()) {
    cerrLock.lock();
    mpidbg << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " external deps: " << externalDependencyCount_
           << " internal deps: " << numPendingInternalDependencies << std::endl;
    cerrLock.unlock();
  }

  if (externalDependencyCount_ == 0 && taskGroup->sc_->useInternalDeps() && initiated_ && !task->usesMPI()) {
    taskGroup->mpiCompletedQueueLock_.writeLock();
    if (mpidbg.active()) {
      cerrLock.lock();
      mpidbg << Parallel::getMPIRank() << " Task " << this->getTask()->getName()
             << " MPI requirements satisfied, placing into external ready queue" << std::endl;
      cerrLock.unlock();
    }

    if (externallyReady_ == false) {
      taskGroup->mpiCompletedTasks_.push(this);
      externallyReady_ = true;
    }
    taskGroup->mpiCompletedQueueLock_.writeUnlock();
  }
}

//_____________________________________________________________________________
//
void
DetailedTask::resetDependencyCounts()
{
  externalDependencyCount_ = 0;
  externallyReady_ = false;
  initiated_ = false;
}

//_____________________________________________________________________________
//
void
DetailedTask::addInternalDependency(       DetailedTask* prerequisiteTask,
                                     const VarLabel*     var )
{
  if (taskGroup->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    std::map<DetailedTask*, InternalDependency*>::iterator foundIt = prerequisiteTask->internalDependents.find(this);
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

void
DetailedTask::assignDevice( int device )
{
  deviceNum_= device;
}

int
DetailedTask::getDeviceNum() const
{
  return deviceNum_;
}

cudaStream_t*
DetailedTask::getCUDAStream() const
{
  return d_cudaStream;
}

void
DetailedTask::setCUDAStream( cudaStream_t* stream )
{
  d_cudaStream = stream;
}

bool
DetailedTask::queryCUDAStreamCompletion()
{
  cudaError_t retVal;
  CUDA_RT_SAFE_CALL( retVal = cudaSetDevice(deviceNum_) );
  retVal = cudaStreamQuery(*d_cudaStream);
  if (retVal == cudaSuccess) {
    return true;
  }
  else if (retVal == cudaErrorNotReady) {
    return false;
  }
  else if (retVal == cudaErrorLaunchFailure) {
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task:"+ getName() , __FILE__, __LINE__));
    return false;
  }
  else {  //other error
    CUDA_RT_SAFE_CALL( retVal );
    return false;
  }
}

#endif // HAVE_CUDA

//_____________________________________________________________________________
//
void
DetailedTask::done( std::vector<OnDemandDataWarehouseP>& dws )
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "This: " << this << " is done with task: " << task << "\n";
    mixedDebug << "Name is: " << task->getName() << " which has (" << internalDependents.size() << ") tasks waiting on it:\n";
    cerrLock.unlock();
  }

  int cnt = 1000;
  map<DetailedTask*, InternalDependency*>::iterator iter;
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
  internalDependencyLock.lock();
  ASSERT(numPendingInternalDependencies > 0);
  unsigned long currentGeneration = taskGroup->getCurrentDependencyGeneration();

  // if false, then the dependency has already been satisfied
  ASSERT(dep->satisfiedGeneration < currentGeneration);

  dep->satisfiedGeneration = currentGeneration;
  numPendingInternalDependencies--;

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << *(dep->dependentTask->getTask()) << " has " << numPendingInternalDependencies << " left.\n";
    cerrLock.unlock();
  }

  if (internaldbg.active()) {
    internaldbg << Parallel::getMPIRank() << " satisfying dependency: prereq: " << *dep->prerequisiteTask << " dep: "
                << *dep->dependentTask << " numPending: " << numPendingInternalDependencies << std::endl;
  }

  if (numPendingInternalDependencies == 0) {
    taskGroup->internalDependenciesSatisfied(this);
    // reset for next timestep
    numPendingInternalDependencies = internalDependencies.size();
  }
  internalDependencyLock.unlock();
}

namespace Uintah {

ostream&
operator<<(       ostream&      out,
            const DetailedTask& task )
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
    else if (patches->size() > 1) {
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
  return out;
}

ostream&
operator<<(       ostream&     out,
            const DetailedDep& dep )
{
  out << dep.req->var->getName();
  if (dep.isNonDataDependency()) {
    out << " non-data dependency";
  }
  else {
    out << " on patch " << dep.fromPatch->getID();
  }
  out << ", matl " << dep.matl << ", low=" << dep.low << ", high=" << dep.high;
  return out;
}

} // end namespace Uintah

//_____________________________________________________________________________
//
void
DetailedTasks::internalDependenciesSatisfied( DetailedTask* task )
{
  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "Begin internalDependenciesSatisfied\n";
    cerrLock.unlock();
  }
  readyQueueLock_.writeLock();
  {
    readyTasks_.push(task);

    if (mixedDebug.active()) {
      cerrLock.lock();
      mixedDebug << *task << " satisfied.  Now " << readyTasks_.size() << " ready.\n";
      cerrLock.unlock();
    }
  }
  readyQueueLock_.writeUnlock();
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  readyQueueLock_.writeLock();
  {
    if (!readyTasks_.empty()) {
      nextTask = readyTasks_.front();
      readyTasks_.pop();
    }
  }
  readyQueueLock_.writeUnlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numInternalReadyTasks()
{
  int size = 0;
  readyQueueLock_.readLock();
  {
    size = readyTasks_.size();
  }
  readyQueueLock_.readUnlock();
  return size;
}

//_____________________________________________________________________________
//
DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  DetailedTask* nextTask = NULL;
  mpiCompletedQueueLock_.writeLock();
  {
    if (!mpiCompletedTasks_.empty()) {
      nextTask = mpiCompletedTasks_.top();
      mpiCompletedTasks_.pop();
    }
  }
  mpiCompletedQueueLock_.writeUnlock();
  return nextTask;
}

//_____________________________________________________________________________
//
int
DetailedTasks::numExternalReadyTasks()
{
  int size = 0;
  mpiCompletedQueueLock_.readLock();
  {
    size = mpiCompletedTasks_.size();
  }
  mpiCompletedQueueLock_.readUnlock();
  return size;
}

#ifdef HAVE_CUDA
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
DependencyBatch::reset()
{
  if (toTasks.size() > 1) {
    if (lock_ == 0) {
      lock_ = scinew Mutex("DependencyBatch receive lock");
    }
  }
  received_ = false;
  madeMPIRequest_ = false;
}

//_____________________________________________________________________________
//
bool
DependencyBatch::makeMPIRequest()
{
  if (toTasks.size() > 1) {
    ASSERT(lock_ != 0);
    if (!madeMPIRequest_) {
      lock_->lock();
      if (!madeMPIRequest_) {
        madeMPIRequest_ = true;
        lock_->unlock();
        return true;  // first to make the request
      }
      else {
        lock_->unlock();
        return false;  // got beat out -- request already made
      }
    }
    return false;  // request already made
  }
  else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(!madeMPIRequest_);
    madeMPIRequest_ = true;
    return true;
  }
}

//_____________________________________________________________________________
//
void
DependencyBatch::addReceiveListener( int mpiSignal )
{
  ASSERT(toTasks.size() > 1);  // only needed when multiple tasks need a batch
  ASSERT(lock_ != 0);
  lock_->lock();
  {
    receiveListeners_.insert(mpiSignal);
  }
  lock_->unlock();
}

//_____________________________________________________________________________
//
void
DependencyBatch::received( const ProcessorGroup * pg )
{
  received_ = true;

  if (mixedDebug.active()) {
    cerrLock.lock();
    mixedDebug << "Received batch message " << messageTag << " from task " << *fromTask << "\n";

    for (DetailedDep* dep = head; dep != 0; dep = dep->next) {
      mixedDebug << "\tSatisfying " << *dep << "\n";
    }
    cerrLock.unlock();
  }
  if (waitout.active()) {
    //add the time waiting on MPI to the wait times per from task
    waittimes[fromTask->getTask()->getName()] += CommRecMPI::WaitTimePerMessage;
  }
  //set all the toVars to valid, meaning the mpi has been completed
  for (vector<Variable*>::iterator iter = toVars.begin(); iter != toVars.end(); iter++) {
    (*iter)->setValid();
  }
  for (list<DetailedTask*>::iterator iter = toTasks.begin(); iter != toTasks.end(); iter++) {
    // if the count is 0, the task will add itself to the external ready queue
    //cout << pg->myrank() << "  Dec: " << *fromTask << " for " << *(*iter) << endl;
    (*iter)->decrementExternalDepCount();
    //cout << Parallel::getMPIRank() << "   task " << **(iter) << " received a message, remaining count " << (*iter)->getExternalDepCount() << endl;
    (*iter)->checkExternalDepCount();
  }

  //clear the variables that have outstanding MPI as they are completed now.
  toVars.clear();

  // TODO APH - Figure this out and clean up (01/31/15)
#if 0
  if (!receiveListeners_.empty()) {
    // only needed when multiple tasks need a batch
    ASSERT(toTasks.size() > 1);
    ASSERT(lock_ != 0);
    lock_->lock();
    {
      for (set<int>::iterator iter = receiveListeners_.begin(); iter != receiveListeners_.end(); ++iter) {
        // send WakeUp messages to threads on the same processor
        MPI_Send(0, 0, MPI_INT, pg->myrank(), *iter, pg->getComm());
      }
      receiveListeners_.clear();
    }
    lock_->unlock();
  }
#endif
}

//_____________________________________________________________________________
//
void
DetailedTasks::logMemoryUse(       std::ostream&  out,
                                   unsigned long& total,
                             const std::string&   tag )
{
  ostringstream elems1;
  elems1 << tasks_.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1, elems1.str(), tasks_.size() * sizeof(DetailedTask), 0);
  ostringstream elems2;
  elems2 << batches_.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1, elems2.str(), batches_.size() * sizeof(DependencyBatch), 0);
  int ndeps = 0;
  for (int i = 0; i < (int)batches_.size(); i++) {
    for (DetailedDep* p = batches_[i]->head; p != 0; p = p->next) {
      ndeps++;
    }
  }
  ostringstream elems3;
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
  map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = reqs.begin(); req_iter != reqs.end(); req_iter++) {
    DetailedTask* fromTask = (*req_iter).first->fromTask;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  list<InternalDependency>::iterator iter;
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

    PatchIDIterator(const vector<const Patch*>::const_iterator& iter)
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
    vector<const Patch*>::const_iterator iter_;
};

//_____________________________________________________________________________
//
string
DetailedTask::getName() const
{
  if (name_ != "") {
    return name_;
  }

  name_ = string(task->getName());

  if (patches != 0) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder(PatchIDIterator(patches->getVector().begin()), PatchIDIterator(patches->getVector().end()));
    name_ += string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if (matls != 0) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder(matls->getVector().begin(), matls->getVector().end());
    name_ += string(" (Matls: ") + matlSet.toString() + ")";
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
