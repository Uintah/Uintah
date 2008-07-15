#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
//#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;
using namespace std;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern UINTAHSHARE SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;
extern DebugStream brydbg;
static DebugStream dbg("TaskGraph", false);
static DebugStream scrubout("Scrubbing", false);
static DebugStream messagedbg("MessageTags", false);
static DebugStream internaldbg("InternalDeps", false);

// for debugging - set the var name to watch one in the scrubout
static string dbgScrubVar = "";
static int dbgScrubPatch = -1;

DetailedTasks::DetailedTasks(SchedulerCommon* sc, const ProcessorGroup* pg,
			     DetailedTasks* first, const TaskGraph* taskgraph,
			     bool mustConsiderInternalDependencies /*= false*/)
  : sc_(sc), d_myworld(pg), first(first), taskgraph_(taskgraph),
    mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
    currentDependencyGeneration_(1),
    extraCommunication_(0),
    readyQueueMutex_("DetailedTasks Ready Queue"),
    readyQueueSemaphore_("Number of Ready DetailedTasks", 0)
{
  int nproc = pg->size();
  stasks_.resize(nproc);
  tasks_.resize(nproc);

  // Set up mappings for the initial send tasks
  int dwmap[Task::TotalDWs];
  for(int i=0;i<Task::TotalDWs;i++)
    dwmap[i]=Task::InvalidDW;
  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = Task::NoDW;
  for(int i=0;i<nproc;i++) {
    stasks_[i]=scinew Task("send old data", Task::InitialSend);
    stasks_[i]->setMapping(dwmap);
    tasks_[i]=scinew DetailedTask(stasks_[i], 0, 0, this);
    tasks_[i]->assignResource(i);
  }
}

DetailedTasks::~DetailedTasks()
{
  for(int i=0;i<(int)batches_.size();i++)
    delete batches_[i];

  for(int i=0;i<(int)tasks_.size();i++)
    delete tasks_[i];

  for(int i=0;i<(int)stasks_.size();i++)
    delete stasks_[i];
}

DependencyBatch::~DependencyBatch()
{
  DetailedDep* dep = head;
  while(dep){
    DetailedDep* tmp = dep->next;
    delete dep;
    dep = tmp;
  }
  delete lock_;
}

void
DetailedTasks::assignMessageTags(int me)
{
  // maps from, to (process) pairs to indices for each batch of that pair
  map< pair<int, int>, int > perPairBatchIndices;

  for(int i=0;i<(int)batches_.size();i++){
    DependencyBatch* batch = batches_[i];
    int from = batch->fromTask->getAssignedResourceIndex();
    ASSERTRANGE(from, 0, d_myworld->size());
    int to = batch->to;
    ASSERTRANGE(to, 0, d_myworld->size());

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing
      // perPairBatchIndices.
      pair<int, int> fromToPair = make_pair(from, to);    
      batches_[i]->messageTag = ++perPairBatchIndices[fromToPair]; /* start with
								     one */
      if (messagedbg.active())
        messagedbg << me << " assigning message num " << batch->messageTag << " from task " << batch->fromTask->getName() << " to task " << batch->toTasks.front()->getName()
                   << ", process " << from << " to process " << to << "\n";
    }
  }
  
  if(dbg.active()) {
    map< pair<int, int>, int >::iterator iter;
    for (iter = perPairBatchIndices.begin(); iter != perPairBatchIndices.end();
	 iter++) {
      int from = iter->first.first;
      int to = iter->first.second;
      int num = iter->second;
      dbg << num << " messages from process " << from 
          << " to process " << to << "\n";
    }
  }
} // end assignMessageTags()

void
DetailedTasks::add(DetailedTask* task)
{
  tasks_.push_back(task);
}

void
DetailedTasks::computeLocalTasks(int me)
{
  initiallyReadyTasks_ = TaskQueue();
  if(localtasks_.size() != 0)
    return;
  for(int i=0;i<(int)tasks_.size();i++){
    DetailedTask* task = tasks_[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
    if(task->getAssignedResourceIndex() == me
       || task->getTask()->getType() == Task::Reduction) {
      localtasks_.push_back(task);

      if (task->areInternalDependenciesSatisfied()) {
	initiallyReadyTasks_.push(task);

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "Initially Ready Task: " 
		     << task->getTask()->getName() << "\n";
	  cerrLock.unlock();
	}
      }
    }
  }
}

DetailedTask::DetailedTask(Task* task, const PatchSubset* patches,
			   const MaterialSubset* matls, DetailedTasks* taskGroup)
  : task(task), patches(patches), matls(matls), comp_head(0),
    taskGroup(taskGroup),
    numPendingInternalDependencies(0),
    internalDependencyLock("DetailedTask Internal Dependencies"),
    resourceIndex(-1)
{
  if(patches) {
    // patches and matls must be sorted
    ASSERT(is_sorted(patches->getVector().begin(), patches->getVector().end(),
		     Patch::Compare()));
    patches->addReference();
  }
  if(matls) {
    // patches and matls must be sorted
    ASSERT(is_sorted(matls->getVector().begin(), matls->getVector().end()));    
    matls->addReference();
  }
}

DetailedTask::~DetailedTask()
{
  if(patches && patches->removeReference())
    delete patches;
  if(matls && matls->removeReference())
    delete matls;
}

void
DetailedTask::doit(const ProcessorGroup* pg,
		   vector<OnDemandDataWarehouseP>& oddws,
		   vector<DataWarehouseP>& dws)
{
  TAU_PROFILE("DetailedTask::doit", " ", TAU_USER); 
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "DetailedTask " << this << " begin doit()\n";
    mixedDebug << " task is " << task << "\n";
    mixedDebug << "   num Pending Deps: " << numPendingInternalDependencies << "\n";
    mixedDebug << "   Originally needed deps (" << internalDependencies.size()
	 << "):\n";

    list<InternalDependency>::iterator iter = internalDependencies.begin();

    for( int i = 0; iter != internalDependencies.end(); iter++, i++ )
      {
	mixedDebug << i << ":    " << *((*iter).prerequisiteTask->getTask()) << "\n";
      }
    cerrLock.unlock();
  }
  for(int i=0;i<(int)dws.size();i++){
    if(oddws[i] != 0)
      oddws[i]->pushRunningTask(task, &oddws);
  }
  task->doit(pg, patches, matls, dws);
  for(int i=0;i<(int)dws.size();i++){
    if(oddws[i] != 0){
      oddws[i]->checkTasksAccesses(patches, matls);
      oddws[i]->popRunningTask();
    }
  }
}

void DetailedTasks::initializeScrubs(vector<OnDemandDataWarehouseP>& dws, int dwmap[])
{
  vector<bool> initialized(dws.size(),false);
  if(0&&scrubout.active())
    scrubout << Parallel::getMPIRank() << " Begin initialize scrubs\n";
  for(int i=0;i<(int)Task::TotalDWs;i++){
    if (dwmap[i] < 0)
      continue;
    OnDemandDataWarehouse* dw = dws[dwmap[i]].get_rep();
    if(dw != 0 && dw->getScrubMode() == DataWarehouse::ScrubComplete){
      // only a OldDW or a CoarseOldDW will have scrubComplete 
      //   But we know a future taskgraph (in a w-cycle) will need the vars if there are fine dws 
      //   between New and Old.  In this case, the scrub count needs to be complemented with CoarseOldDW
      int tgtype = getTaskGraph()->getType();
      if (!initialized[dwmap[i]] || tgtype == Scheduler::IntermediateTaskGraph) {
        // if we're intermediate, we're going to need to make sure we don't scrub CoarseOld before we finish using it
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << dw->getID() << " for DW type " << i << " ADD=" << initialized[dwmap[i]] << '\n';
        dw->initializeScrubs(i, &(first?first->scrubCountTable_:scrubCountTable_), initialized[dwmap[i]]);
      }
      if (i != Task::OldDW && tgtype != Scheduler::IntermediateTaskGraph && dwmap[Task::NewDW] - dwmap[Task::OldDW] > 1) {
        // add the CoarseOldDW's scrubs to the OldDW, so we keep it around for future task graphs
        OnDemandDataWarehouse* olddw = dws[dwmap[Task::OldDW]].get_rep();
        scrubout << Parallel::getMPIRank() << " Initializing scrubs on dw: " << olddw->getID() << " for DW type " << i << " ADD=" << 1 << '\n';
        ASSERT(initialized[dwmap[Task::OldDW]]);
        olddw->initializeScrubs(i, &(first?first->scrubCountTable_:scrubCountTable_), true);
      }
      initialized[dwmap[i]] = true;
    }
  }
  if(0&&scrubout.active())
    scrubout << Parallel::getMPIRank() << " End initialize scrubs\n";
}

void
DetailedTask::scrub(vector<OnDemandDataWarehouseP>& dws)
{
  const Task* task = getTask();

  if(0&&scrubout.active())
    scrubout << Parallel::getMPIRank() << " Starting scrub after task: " << *this << '\n';
  const set<const VarLabel*, VarLabel::Compare>& initialRequires
    = taskGroup->getSchedulerCommon()->getInitialRequiredVars();
  const set<string>& unscrubbables = taskGroup->getSchedulerCommon()->getNoScrubVars();

  // Decrement the scrub count for each of the required variables
  for(const Task::Dependency* req = task->getRequires();
      req != 0; req=req->next){
    TypeDescription::Type type = req->var->typeDescription()->getType();
    Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
    if(type != TypeDescription::ReductionVariable && 
       type !=TypeDescription::SoleVariable){
      int dw = req->mapDataWarehouse();
      
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if(scrubmode == DataWarehouse::ScrubComplete ||
	 (scrubmode == DataWarehouse::ScrubNonPermanent &&
	  initialRequires.find(req->var) == initialRequires.end())){

        if (unscrubbables.find(req->var->getName()) != unscrubbables.end())
          continue;

	constHandle<PatchSubset> patches = req->getPatchesUnderDomain(getPatches());
	constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(getMaterials());
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  Patch::selectType neighbors;
	  IntVector low, high;
          
          if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel || req->numGhostCells == 0){
            // we already have the right patches
            neighbors.push_back(patch);
          }
          else {
            patch->computeVariableExtents(type, req->var->getBoundaryLayer(),
                                          req->gtype, req->numGhostCells,
                                          neighbors, low, high);
          }
	  for(int i=0;i<neighbors.size();i++){
	    const Patch* neighbor=neighbors[i];

            if (patch->getLevel()->getIndex() > 0 && patch != neighbor && req->patches_dom == Task::NormalDomain) {
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
              IntVector l = patch->getExtraLowIndex(basis, IntVector(0,0,0)), h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
              IntVector fl = neighbor->getLowIndex(basis), fh = neighbor->getHighIndex(basis);
              IntVector il = Max(l, neighbor->getLevel()->mapCellToCoarser(fl));
              IntVector ih = Min(h, neighbor->getLevel()->mapCellToCoarser(fh));
              if (ih.x() <= il.x() || ih.y() <= il.y() || ih.z() <= il.z()) {
                continue;
              }
            }
	    for (int m=0;m<matls->size();m++){
              int count;
              try {
                // there are a few rare cases in an AMR framework where you require from an OldDW, but only
                // ones internal to the W-cycle (and not the previous timestep) which can have variables not exist in the OldDW.
                if (dws[dw]->exists(req->var, matls->get(m), neighbor)) {
                  count = dws[dw]->decrementScrubCount(req->var, matls->get(m), neighbor);
                  if(scrubout.active() && 
                     (req->var->getName() == dbgScrubVar || dbgScrubVar == "") && 
                     (neighbor->getID() == dbgScrubPatch || dbgScrubPatch == -1)){
                    scrubout << Parallel::getMPIRank() << "   decrementing scrub count for requires of " << dws[dw]->getID() << "/" << neighbor->getID() << "/" << matls->get(m) << "/" << req->var->getName() << ": " << count << (count == 0?" - scrubbed\n":"\n");
                  }
                }
              } catch (UnknownVariable& e) {
                cout << "   BAD BOY FROM Task : " << *this << " scrubbing " << *req << " PATCHES: " << *patches.get_rep() << endl;
                throw e;
              }
	    }
	  }
	}
      }
    }
  }

  // Scrub modifies
  for(const Task::Dependency* mod = task->getModifies(); mod != 0; mod=mod->next){
    int dw = mod->mapDataWarehouse();
    DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
    if(scrubmode == DataWarehouse::ScrubComplete ||
       (scrubmode == DataWarehouse::ScrubNonPermanent &&
	initialRequires.find(mod->var) == initialRequires.end())){
      
      if (unscrubbables.find(mod->var->getName()) != unscrubbables.end())
        continue;

      constHandle<PatchSubset> patches = mod->getPatchesUnderDomain(getPatches());
      constHandle<MaterialSubset> matls = mod->getMaterialsUnderDomain(getMaterials());
      TypeDescription::Type type = mod->var->typeDescription()->getType();
      if(type != TypeDescription::ReductionVariable && 
         type != TypeDescription::SoleVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++){
	    int count = dws[dw]->decrementScrubCount(mod->var, matls->get(m), patch);
	    if(scrubout.active() && 
               (mod->var->getName() == dbgScrubVar || dbgScrubVar == "") && 
               (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1))
	      scrubout << Parallel::getMPIRank() << "   decrementing scrub count for modifies of " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << mod->var->getName() << ": " << count << (count == 0?" - scrubbed\n":"\n");
	  }
	}
      }
    }
  }
  
  // Set the scrub count for each of the computes variables
  for(const Task::Dependency* comp = task->getComputes();
      comp != 0; comp=comp->next){
    TypeDescription::Type type = comp->var->typeDescription()->getType();
    if(type != TypeDescription::ReductionVariable &&
       type != TypeDescription::SoleVariable){
      int whichdw = comp->whichdw;
      int dw = comp->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if(scrubmode == DataWarehouse::ScrubComplete ||
	 (scrubmode == DataWarehouse::ScrubNonPermanent &&
	  initialRequires.find(comp->var) == initialRequires.end())){
	constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(getPatches());
	constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(getMaterials());

        if (unscrubbables.find(comp->var->getName()) != unscrubbables.end())
          continue;

	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++){
	    int matl = matls->get(m);
	    int count;
	    if(taskGroup->getScrubCount(comp->var, matl, patch, whichdw, count)){
	      if(scrubout.active() && 
                 (comp->var->getName() == dbgScrubVar || dbgScrubVar == "") &&
                 (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1))
		scrubout << Parallel::getMPIRank() << "   setting scrub count for computes of " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << ": " << count << '\n';
	      dws[dw]->setScrubCount(comp->var, matl, patch, count);
	    } else {
	      // Not in the scrub map, must be never needed...
	      if(scrubout.active() && 
                 (comp->var->getName() == dbgScrubVar || dbgScrubVar == "") &&
                 (patch->getID() == dbgScrubPatch || dbgScrubPatch == -1))
		scrubout << Parallel::getMPIRank() << "   trashing variable immediately after compute: " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << '\n';
	      dws[dw]->scrub(comp->var, matl, patch);
	    }
	  }
	}
      }
    }
  }
}

// used to be in terms of the dw index within the scheduler,
// but now store WhichDW.  This enables multiple tg execution
void DetailedTasks::addScrubCount(const VarLabel* var, int matlindex,
                                  const Patch* patch, int dw)
{
  if(patch->isVirtual())
    patch = patch->getRealPatch();
  ScrubItem key(var, matlindex, patch, dw);
  ScrubItem* result;
  result = (first?first->scrubCountTable_:scrubCountTable_).lookup(&key);
  if(!result){
    result = ::new ScrubItem(var, matlindex, patch, dw);
    (first?first->scrubCountTable_:scrubCountTable_).insert(result);
  }
  result->count++;
  if (scrubout.active() && (var->getName() == dbgScrubVar || dbgScrubVar == "") && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1))
    scrubout << Parallel::getMPIRank() << " Adding Scrub count for req of " << dw << "/" << patch->getID() << "/" << matlindex << "/" << *var << ": " << result->count << endl;
}

void DetailedTasks::setScrubCount(const Task::Dependency* req, int matl, const Patch* patch,
                                  vector<OnDemandDataWarehouseP>& dws)
{
  ASSERT(!patch->isVirtual());
  DataWarehouse::ScrubMode scrubmode = dws[req->mapDataWarehouse()]->getScrubMode();
  const set<const VarLabel*, VarLabel::Compare>& initialRequires
    = getSchedulerCommon()->getInitialRequiredVars();
  if(scrubmode == DataWarehouse::ScrubComplete ||
     (scrubmode == DataWarehouse::ScrubNonPermanent &&
      initialRequires.find(req->var) == initialRequires.end())){
    int scrubcount;
    if(!getScrubCount(req->var, matl, patch, req->whichdw, scrubcount)){
      SCI_THROW(InternalError("No scrub count for received MPIVariable: "+req->var->getName(), __FILE__, __LINE__));
    }
    if(scrubout.active() && (req->var->getName() == dbgScrubVar || dbgScrubVar == "") && (dbgScrubPatch == patch->getID() || dbgScrubPatch == -1))
      scrubout << Parallel::getMPIRank() << " setting scrubcount for recv of " << req->mapDataWarehouse() << "/" << patch->getID() << "/" << matl << "/" << req->var->getName() << ": " << scrubcount << '\n';
    dws[req->mapDataWarehouse()]->setScrubCount(req->var, matl, patch, scrubcount);
  }
}

bool DetailedTasks::getScrubCount(const VarLabel* label, int matlIndex,
				  const Patch* patch, int dw, int& count)
{
  ASSERT(!patch->isVirtual());
  ScrubItem key(label, matlIndex, patch, dw);
  ScrubItem* result = (first?first->scrubCountTable_:scrubCountTable_).lookup(&key);
  if(result){
    count = result->count;
    return true;
  } else {
    return false;
  }
}
void DetailedTasks::createScrubCounts()
{
  (first?first->scrubCountTable_:scrubCountTable_).remove_all();
  // Go through each of the tasks and determine which variables it will require
  for(int i=0;i<(int)localtasks_.size();i++){
    DetailedTask* dtask = localtasks_[i];
    const Task* task = dtask->getTask();
    for(const Task::Dependency* req = task->getRequires(); req != 0; req=req->next){
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->whichdw;
      TypeDescription::Type type = req->var->typeDescription()->getType();
      if(type != TypeDescription::ReductionVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  Patch::selectType neighbors;
	  IntVector low, high;
	  patch->computeVariableExtents(type,req->var->getBoundaryLayer(),
					req->gtype, req->numGhostCells,
					neighbors, low, high);
	  for(int i=0;i<neighbors.size();i++){
	    const Patch* neighbor=neighbors[i];
	    for (int m=0;m<matls->size();m++)
	      addScrubCount(req->var, matls->get(m), neighbor, whichdw);
	  }
	}
      }
    }
    for(const Task::Dependency* req = task->getModifies(); req != 0; req=req->next){
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int whichdw = req->whichdw;
      TypeDescription::Type type = req->var->typeDescription()->getType();
      if(type != TypeDescription::ReductionVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++)
	    addScrubCount(req->var, matls->get(m), patch, whichdw);
	}
      }
    }
  }
  if(scrubout.active()){
    scrubout << Parallel::getMPIRank() << " scrub counts:\n";
    scrubout << Parallel::getMPIRank() << " DW/Patch/Matl/Label\tCount\n";
    for(FastHashTableIter<ScrubItem> iter(&(first?first->scrubCountTable_:scrubCountTable_));
	iter.ok(); ++iter){
      const ScrubItem* rec = iter.get_key();
      scrubout << rec->dw << '/' << (rec->patch?rec->patch->getID():0) << '/'
	       << rec->matl << '/' <<  rec->label->getName()
	       << "\t\t" << rec->count << '\n';
    }
    scrubout << Parallel::getMPIRank() << " end scrub counts\n";
  }
}

void DetailedTask::findRequiringTasks(const VarLabel* var,
				      list<DetailedTask*>& requiringTasks)
{
  // find requiring tasks

  // find external requires
  for (DependencyBatch* batch = getComputes(); batch != 0;
       batch = batch->comp_next) {
    for (DetailedDep* dep = batch->head; dep != 0; 
	 dep = dep->next) {
      if (dep->req->var == var) {
	requiringTasks.insert(requiringTasks.end(), dep->toTasks.begin(),
			      dep->toTasks.end());
      }
    }
  }

  // find internal requires
  map<DetailedTask*, InternalDependency*>::iterator internalDepIter;
  for (internalDepIter = internalDependents.begin();
       internalDepIter != internalDependents.end(); ++internalDepIter) {
    if (internalDepIter->second->vars.find(var) !=
	internalDepIter->second->vars.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

DetailedDep* DetailedTasks::findMatchingDetailedDep(DependencyBatch* batch, DetailedTask* toTask, Task::Dependency* req, 
                                                    const Patch* fromPatch, int matl, IntVector low, IntVector high,
                                                    IntVector& totalLow, IntVector& totalHigh)
{
  totalLow = low;
  totalHigh = high;
  DetailedDep* dep = batch->head;
  for(;dep != 0; dep = dep->next){
    if(fromPatch == dep->fromPatch && matl == dep->matl
       && (req == dep->req
	   || (req->var->equals(dep->req->var)
	       && req->mapDataWarehouse() == dep->req->mapDataWarehouse()))) {

      // total range - the same var in each dep needs to have the same patchlow/high
      dep->patchLow = totalLow = Min(totalLow, dep->patchLow);
      dep->patchHigh = totalHigh = Max(totalHigh, dep->patchHigh);

      int ngcDiff = req->numGhostCells > dep->req->numGhostCells ? (req->numGhostCells - dep->req->numGhostCells) : 0;
      IntVector new_l = Min(low, dep->low);
      IntVector new_h = Max(high, dep->high);
      IntVector newRange = new_h-new_l;
      IntVector requiredRange = high-low;
      IntVector oldRange = dep->high-dep->low + IntVector(ngcDiff, ngcDiff, ngcDiff);
      int newSize = newRange.x()*newRange.y()*newRange.z();
      int requiredSize = requiredRange.x()*requiredRange.y()*requiredRange.z();
      int oldSize = oldRange.x()*oldRange.y()*oldRange.z();
      
      bool extraComm = newSize > requiredSize+oldSize;

      if (sc_->useSmallMessages()) {
        // If two patches on the same processor want data from the same patch on a different
        // processor, we can either pack them in one dependency and send the min and max of their range (which
        // will frequently result in sending the entire patch), or we can use two dependencies (which will get packed into
        // one message) and only send the resulting data.
        
        // We want to create a new dep in such cases.  However, we don't want to do this in cases where we simply add more
        // ghost cells.
        if (!extraComm)
          break;
        else if (dbg.active()) {
          dbg << d_myworld->myrank() << "            Ignoring: " << dep->low << " " << dep->high << ", fromPatch = ";
          if (fromPatch)
            dbg << fromPatch->getID() << '\n';
          else
            dbg << "NULL\n";
          dbg << d_myworld->myrank() << " TP: " << totalLow << " " << totalHigh << endl;
        }

      }
      else {
        if (extraComm) {
          extraCommunication_ += newSize - (requiredSize+oldSize);
        }
        break;
      }
    }
  }
  return dep;
}

void
DetailedTasks::possiblyCreateDependency(DetailedTask* from,
					Task::Dependency* comp,
					const Patch* fromPatch,
					DetailedTask* to,
					Task::Dependency* req,
					const Patch */*toPatch*/,
					int matl,
					const IntVector& low,
					const IntVector& high,
                                        DetailedDep::CommCondition cond)
{
  // TODO - maybe still create internal depencies for threaded scheduler?
  // TODO - perhaps move at least some of this to TaskGraph?
  ASSERTRANGE(from->getAssignedResourceIndex(), 0, d_myworld->size());
  ASSERTRANGE(to->getAssignedResourceIndex(), 0, d_myworld->size());
  if(dbg.active()) {
    dbg << d_myworld->myrank() << "          " << *to << " depends on " << *from << "\n";
    if(comp)
      dbg << d_myworld->myrank() << "            From comp " << *comp;
    else
      dbg << d_myworld->myrank() << "            From OldDW ";
    dbg << " to req " << *req << '\n';
  }

  int toresource = to->getAssignedResourceIndex();
  int fromresource = from->getAssignedResourceIndex();

  if ((toresource == d_myworld->myrank() || 
       (req->patches_dom != Task::NormalDomain && fromresource == d_myworld->myrank())) && 
       fromPatch && !req->var->typeDescription()->isReductionVariable()) {
    // add scrub counts for local tasks, and not for non-data deps
    addScrubCount(req->var, matl, fromPatch, req->whichdw);
  }

  if(fromresource == d_myworld->myrank() && (fromresource == toresource || 
     req->var->typeDescription()->isReductionVariable())) {
    to->addInternalDependency(from, req->var);
    return;
  }

  // if neither task talks to this processor, return
  if (fromresource != d_myworld->myrank() && toresource != d_myworld->myrank()) {
    return;
  }


  DependencyBatch* batch = from->getComputes();
  for(;batch != 0; batch = batch->comp_next){
    if(batch->to == toresource)
      break;
  }
  if(!batch){
    batch = scinew DependencyBatch(toresource, from, to);
    batches_.push_back(batch);
    from->addComputes(batch);
#if SCI_ASSERTION_LEVEL >= 2
    bool newRequireBatch = to->addRequires(batch);
#else
    to->addRequires(batch);
#endif
    ASSERTL2(newRequireBatch);
    if(dbg.active())
      dbg << d_myworld->myrank() << "          NEW BATCH!\n";
  } else if (mustConsiderInternalDependencies_) { // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->toTasks.push_back(to);
    }
    if(dbg.active())
      dbg << d_myworld->myrank() << "          USING PREVIOUSLY CREATED BATCH!\n";
  }
  // the total region spanned by the dep(s)
  IntVector varRangeLow(INT_MAX, INT_MAX, INT_MAX), varRangeHigh(INT_MIN, INT_MIN, INT_MIN);
  DetailedDep* dep = findMatchingDetailedDep(batch, to, req, fromPatch, matl, low, high, varRangeLow, varRangeHigh);

  if(!dep){
    dep = scinew DetailedDep(batch->head, comp, req, to, fromPatch, matl, 
			     low, high, cond);
    // these are to post all the particle quantities up front - sort them in TG::createDetailedDepenedencies
    if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
      if (fromresource == d_myworld->myrank())
        particleSends_[toresource].insert(PSPatchMatlGhost(fromPatch, matl, low, high, (int) cond));
      else if (toresource == d_myworld->myrank())
        particleRecvs_[fromresource].insert(PSPatchMatlGhost(fromPatch, matl, low, high, (int) cond));
      if (req->var->getName() == "p.x") 
        dbg << d_myworld->myrank() << " scheduling particles from " << fromresource << " to " << toresource 
            << " on patch " << fromPatch->getID() << " matl " << matl << " range " << low << " " << high 
            << " cond " << cond << " dw " << req->mapDataWarehouse() << endl;
    }
    
    batch->head = dep;
    if(dbg.active()) {
      dbg << d_myworld->myrank() << "            ADDED " << low << " " << high << ", fromPatch = ";
      if (fromPatch)
	dbg << fromPatch->getID() << '\n';
      else
	dbg << "NULL\n";	
    }
  }
  else {
    IntVector l = Min(low, dep->low), h = Max(high, dep->high);
    dep->toTasks.push_back(to);
    if(dbg.active()){
      dbg << d_myworld->myrank() << "            EXTENDED from " << dep->low << " " << dep->high << " to " << l << " " << h << "\n";
      dbg << *req->var << '\n';
      dbg << *dep->req->var << '\n';
      if(comp)
        dbg << *comp->var << '\n';
      if(dep->comp)
        dbg << *dep->comp->var << '\n';
    }
    // extend these too
    if (req->var->typeDescription()->getType() == TypeDescription::ParticleVariable && req->whichdw == Task::OldDW) {
      PSPatchMatlGhost pmg(fromPatch, matl, dep->low, dep->high, (int) cond);
      PSPatchMatlGhost pmg_new(fromPatch, matl, l, h, (int) cond);
      if (req->var->getName() == "p.x")
        dbg << d_myworld->myrank() << " extending particles from " << fromresource << " to " << toresource 
            << " var " << *req->var << " on patch " << fromPatch->getID() << " matl " << matl 
            << " range " << dep->low << " " << dep->high << " cond " << cond 
            << " dw " << req->mapDataWarehouse() << " extended to " << l << " " << h << endl;
      if (fromresource == d_myworld->myrank()) {
        set<PSPatchMatlGhost>::iterator iter= particleSends_[toresource].find(pmg);
        // if it is not there, assume it already got extended
        if (iter != particleSends_[toresource].end()) {
          particleSends_[toresource].erase(pmg);
          particleSends_[toresource].insert(pmg_new);
        }
      }
      else if (toresource == d_myworld->myrank()) {
        set<PSPatchMatlGhost>::iterator iter= particleRecvs_[fromresource].find(pmg);
        if (iter != particleRecvs_[toresource].end()) {
          particleRecvs_[fromresource].erase(pmg);
          particleRecvs_[fromresource].insert(pmg_new);
        }
      }
    }
    dep->low=l;
    dep->high=h;  
  }
  // the total range of my dep and any deps later in the list with the same var/fromPatch/matl/dw
  // (to set the next one, which will be the head of the list, you only need to see the following one)
  dep->patchLow = varRangeLow;
  dep->patchHigh = varRangeHigh;
}

DetailedTask*
DetailedTasks::getOldDWSendTask(int proc)
{
  // These are the first N tasks
  return tasks_[proc];
}

void
DetailedTask::addComputes(DependencyBatch* comp)
{
  comp->comp_next=comp_head;
  comp_head=comp;
}

bool
DetailedTask::addRequires(DependencyBatch* req)
{
  // return true if it is adding a new batch
  return reqs.insert(make_pair(req, req)).second;
}

// can be called in one of two places - when the last MPI Recv has completed, or from MPIScheduler
void DetailedTask::checkExternalDepCount()
{
  //cout << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " ext deps: " << externalDependencyCount_ << " int deps: " << numPendingInternalDependencies << endl;
  if (externalDependencyCount_ == 0 && taskGroup->sc_->useInternalDeps() && initiated_ && externallyReady_ == false) {
    //cout << Parallel::getMPIRank() << " Task " << this->getTask()->getName() << " ready\n";
    taskGroup->mpiCompletedTasks_.push(this);
    externallyReady_ = true;
  }
}

void DetailedTask::resetDependencyCounts()
{
  externalDependencyCount_ = 0;
  externallyReady_ = false;
  initiated_ = false;
}

void
DetailedTask::addInternalDependency(DetailedTask* prerequisiteTask,
				    const VarLabel* var)
{
  if (taskGroup->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    map<DetailedTask*, InternalDependency*>::iterator foundIt =
      prerequisiteTask->internalDependents.find(this);
    if (foundIt == prerequisiteTask->internalDependents.end()) {
      internalDependencies.push_back(InternalDependency(prerequisiteTask,
							this, var,
							0/* not satisfied */));
      prerequisiteTask->
	internalDependents[this] = &internalDependencies.back();
      numPendingInternalDependencies = internalDependencies.size();
      if(internaldbg.active() ) {
        internaldbg << Parallel::getMPIRank() << " Adding dependency between " << *this << " and " << *prerequisiteTask << "\n";
      }

    }
    else {
      foundIt->second->addVarLabel(var);
    }
  }
}

void
DetailedTask::done(vector<OnDemandDataWarehouseP>& dws)
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "This: " << this << " is done with task: " << task << "\n";
    mixedDebug << "Name is: " << task->getName()
	       << " which has (" << internalDependents.size() 
	       << ") tasks waiting on it:\n";
    cerrLock.unlock();
  }

  int cnt = 1000;
  map<DetailedTask*, InternalDependency*>::iterator iter;
  for (iter = internalDependents.begin(); iter != internalDependents.end(); 
       iter++) {
    InternalDependency* dep = (*iter).second;

    if(internaldbg.active() ) {
      internaldbg << Parallel::getMPIRank() << " Depend satisfied between " << *dep->dependentTask << " and " << *this << "\n";
    }

    dep->dependentTask->dependencySatisfied(dep);
    cnt++;
  }
}

void DetailedTask::dependencySatisfied(InternalDependency* dep)
{
 internalDependencyLock.lock();
  ASSERT(numPendingInternalDependencies > 0);
  unsigned long currentGeneration = taskGroup->getCurrentDependencyGeneration();

  // if false, then the dependency has already been satisfied
  ASSERT(dep->satisfiedGeneration < currentGeneration);

  dep->satisfiedGeneration = currentGeneration;
  numPendingInternalDependencies--;

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << *(dep->dependentTask->getTask()) << " has " 
	       << numPendingInternalDependencies << " left.\n";
    cerrLock.unlock();
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
operator<<(ostream& out, const DetailedTask& task)
{
  out << task.getTask()->getName();
  const PatchSubset* patches = task.getPatches();
  if(patches){
    out << ", on patch";
    if(patches->size() > 1)
      out << "es";
    out << " ";
    for(int i=0;i<patches->size();i++){
      if(i>0)
	out << ",";
      out << patches->get(i)->getID();
    }
    // a once-per-proc task is liable to have multiple levels, and thus calls to getLevel(patches) will fail
    if (task.getTask()->getType() == Task::OncePerProc)
      out << ", on multiple levels";
    else if (patches->size() > 1)
      out << ", Level " << getLevel(patches)->getIndex();
  }
  const MaterialSubset* matls = task.getMaterials();
  if(matls){
    out << ", on material";
    if(matls->size() > 1)
      out << "s";
    out << " ";
    for(int i=0;i<matls->size();i++){
      if(i>0)
	out << ",";
      out << matls->get(i);
    }
  }
  out << ", resource ";
  if(task.getAssignedResourceIndex() == -1)
    out << "unassigned";
  else
    out << task.getAssignedResourceIndex();
  return out;
}

ostream&
operator<<(ostream& out, const DetailedDep& dep)
{
  out << dep.req->var->getName();
  if (dep.isNonDataDependency())
    out << " non-data dependency";
  else
    out << " on patch " << dep.fromPatch->getID();
  out << ", matl " << dep.matl << ", low=" << dep.low << ", high=" << dep.high;
  return out;
}
}

void
DetailedTasks::internalDependenciesSatisfied(DetailedTask* task)
{
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Begin internalDependenciesSatisfied\n";
    cerrLock.unlock();
  }
#if !defined( _AIX )
  readyQueueMutex_.lock();
#endif

  readyTasks_.push(task);

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << *task << " satisfied.  Now " 
	       << readyTasks_.size() << " ready.\n";
    cerrLock.unlock();
  }
#if !defined( _AIX )
  // need to make a non-binary semaphore under aix for this to work.
  readyQueueSemaphore_.up();
  readyQueueMutex_.unlock();
#endif
}

DetailedTask*
DetailedTasks::getNextInternalReadyTask()
{
  // Block until the list has an item in it.
#if !defined( _AIX )
  readyQueueSemaphore_.down();
  readyQueueMutex_.lock();
#endif
  DetailedTask* nextTask = readyTasks_.front();
  readyTasks_.pop();
#if !defined( _AIX )
  readyQueueMutex_.unlock();
#endif
  return nextTask;
}

DetailedTask*
DetailedTasks::getNextExternalReadyTask()
{
  DetailedTask* nextTask = mpiCompletedTasks_.front();
  mpiCompletedTasks_.pop();
  //cout << Parallel::getMPIRank() << "    Getting: " << *nextTask << "  new size: " << mpiCompletedTasks_.size() << endl;
  return nextTask;
}

void
DetailedTasks::initTimestep()
{
  readyTasks_ = initiallyReadyTasks_;
#if !defined( _AIX )
  readyQueueSemaphore_.up((int)readyTasks_.size());
#endif
  incrementDependencyGeneration();
  initializeBatches();
}

void DetailedTasks::incrementDependencyGeneration()
{
  if (currentDependencyGeneration_ >= ULONG_MAX)
    SCI_THROW(InternalError("DetailedTasks::currentDependencySatisfyingGeneration has overflowed", __FILE__, __LINE__));
  currentDependencyGeneration_++;
}

void DetailedTasks::initializeBatches()
{
  for (int i = 0; i < (int)batches_.size(); i++) {
    batches_[i]->reset();
  }
}

void DependencyBatch::reset()
{
  if (toTasks.size() > 1) {
    if (lock_ == 0)
      lock_ = scinew Mutex("DependencyBatch receive lock");
  }
  received_ = false;
  madeMPIRequest_ = false; 
}

bool DependencyBatch::makeMPIRequest()
{
  if (toTasks.size() > 1) {
    ASSERT(lock_ != 0);    
    if (!madeMPIRequest_) {
      lock_->lock();
      if (!madeMPIRequest_) {
	madeMPIRequest_ = true;
	lock_->unlock();
	return true; // first to make the request
      }
      else {
	lock_->unlock();
	return false; // got beat out -- request already made
      }
    }
    return false; // request already made
  }
  else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(!madeMPIRequest_);
    madeMPIRequest_ = true;
    return true;
  }
}

void DependencyBatch::addReceiveListener(int mpiSignal)
{
  ASSERT(toTasks.size() > 1); // only needed when multiple tasks need a batch
  ASSERT(lock_ != 0);  
 lock_->lock();
  receiveListeners_.insert(mpiSignal);
 lock_->unlock();
}

void DependencyBatch::received(const ProcessorGroup * pg)
{
  received_ = true;
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Received batch message " << messageTag 
	       << " from task " << *fromTask << "\n";
    
    for (DetailedDep* dep = head; dep != 0; dep = dep->next)
      mixedDebug << "\tSatisfying " << *dep << "\n";
    cerrLock.unlock();
  }

  list<DetailedTask*>::iterator iter;
  for (iter = toTasks.begin(); iter != toTasks.end(); iter++) {
    // if the count is 0, the task will add itself to the external ready queue
    //cout << pg->myrank() << "  Dec: " << *fromTask << " for " << *(*iter) << endl;
    (*iter)->decrementExternalDepCount();
    //cout << Parallel::getMPIRank() << "   task " << **(iter) << " received a message, remaining count " << (*iter)->getExternalDepCount() << endl;
    (*iter)->checkExternalDepCount();
  }

#if 0
  if (!receiveListeners_.empty()) {
    // only needed when multiple tasks need a batch
    ASSERT(toTasks.size() > 1);
    ASSERT(lock_ != 0);        
   lock_->lock();
    for (set<int>::iterator iter = receiveListeners_.begin();
	 iter != receiveListeners_.end(); ++iter) {
      // send WakeUp messages to threads on the same processor
      MPI_Send(0, 0, MPI_INT, pg->myrank(), *iter, pg->getComm());
    }
    receiveListeners_.clear();
   lock_->unlock();
  }
#endif
}

void DetailedTasks::logMemoryUse(ostream& out, unsigned long& total,
				 const std::string& tag)
{
  ostringstream elems1;
  elems1 << tasks_.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1,
	    elems1.str(), tasks_.size()*sizeof(DetailedTask), 0);
  ostringstream elems2;
  elems2 << batches_.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1,
	    elems2.str(), batches_.size()*sizeof(DependencyBatch), 0);
  int ndeps=0;
  for(int i=0;i<(int)batches_.size();i++){
    for(DetailedDep* p=batches_[i]->head; p != 0; p = p->next)
      ndeps++;
  }
  ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "DetailedDep", 0, -1,
	    elems3.str(), ndeps*sizeof(DetailedDep), 0);
}

void DetailedTasks::emitEdges(ProblemSpecP edgesElement, int rank)
{
  for (int i = 0; i < (int)tasks_.size(); i++) {
    ASSERTRANGE(tasks_[i]->getAssignedResourceIndex(), 0, d_myworld->size());
    if (tasks_[i]->getAssignedResourceIndex() == rank) {
      tasks_[i]->emitEdges(edgesElement);
    }
  }
}

void DetailedTask::emitEdges(ProblemSpecP edgesElement)
{
  map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = reqs.begin(); req_iter != reqs.end(); req_iter++) {
    DetailedTask* fromTask = (*req_iter).first->fromTask;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  list<InternalDependency>::iterator iter;
  for (iter = internalDependencies.begin();
       iter != internalDependencies.end(); iter++) {
    DetailedTask* fromTask = (*iter).prerequisiteTask;
    if (getTask()->isReductionTask() &&
	fromTask->getTask()->isReductionTask()) {
      // Ignore internal links between reduction tasks because they
      // are only needed for logistic reasons
      continue;
    }
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }
}

class PatchIDIterator
{
public:
  PatchIDIterator(const vector< const Patch*>::const_iterator& iter)
    : iter_(iter) {}

  PatchIDIterator& operator=(const PatchIDIterator& iter2)
  { iter_ = iter2.iter_; return *this; }
  
  int operator*()
  {
    const Patch* patch = *iter_; //vector<Patch*>::iterator::operator*();
    return patch ? patch->getID() : -1;
  }

  PatchIDIterator& operator++()
  { iter_++; return *this; }

  bool operator!=(const PatchIDIterator& iter2)
  { return iter_ != iter2.iter_; }
  
private:
  vector<const Patch*>::const_iterator iter_;
};

string DetailedTask::getName() const
{
  if (name_ != "")
    return name_;

  name_ = string(task->getName());

  if (patches != 0) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder(PatchIDIterator(patches->getVector().begin()),
			PatchIDIterator(patches->getVector().end()));
    name_ += string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if (matls != 0) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder(matls->getVector().begin(),
		       matls->getVector().end());
    name_ += string(" (Matls: ") + matlSet.toString() + ")";
  }
  
  return name_;
}
