#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <sci_algorithm.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;
using std::is_sorted;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time) 
// From: sus.cc
extern SCIRun::Mutex cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("TaskGraph", false);

const int NO_SCRUB = -1;

DetailedTasks::DetailedTasks(const ProcessorGroup* pg,
			     const TaskGraph* taskgraph,
			     bool scrubNew /*= true */,
			     bool mustConsiderInternalDependencies /*= false*/)
  : taskgraph(taskgraph),
    mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
    currentDependencyGeneration_(1),
    readyQueueMutex_("DetailedTasks Ready Queue"),
    readyQueueSemaphore_("Number of Ready DetailedTasks", 0),
    scrubNew_(scrubNew)
{
  int nproc = pg->size();
  stasks.resize(nproc);
  tasks.resize(nproc);
  for(int i=0;i<nproc;i++) {
    stasks[i]=scinew Task("send old data", Task::InitialSend);
    tasks[i]=scinew DetailedTask(stasks[i], 0, 0, this);
    tasks[i]->assignResource(i);
  }
  scrubExtraneousOldDW_ = false;
}

DetailedTasks::~DetailedTasks()
{
  for(int i=0;i<(int)batches.size();i++)
    delete batches[i];

  for(int i=0;i<(int)tasks.size();i++)
    delete tasks[i];

  for(int i=0;i<(int)stasks.size();i++)
    delete stasks[i];
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
  //delete cv_;
}

void
DetailedTasks::assignMessageTags(int me)
{
  // maps from, to (process) pairs to indices for each batch of that pair
  map< pair<int, int>, int > perPairBatchIndices;

  for(int i=0;i<(int)batches.size();i++){
    DependencyBatch* batch = batches[i];
    int from = batch->fromTask->getAssignedResourceIndex();
    ASSERT(from != -1);
    int to = batch->to;
    ASSERT(to != -1);

    if (from == me || to == me) {
      // Easier to go in reverse order now, instead of reinitializing
      // perPairBatchIndices.
      pair<int, int> fromToPair = make_pair(from, to);    
      batches[i]->messageTag = ++perPairBatchIndices[fromToPair]; /* start with
								     one */
    }
  }
  
  if(dbg.active()) {
    map< pair<int, int>, int >::iterator iter;
    for (iter = perPairBatchIndices.begin(); iter != perPairBatchIndices.end();
	 iter++) {
      int from = iter->first.first;
      int to = iter->first.second;
      int num = iter->second;
      dbg << num << " messages from process " << from << " to process " << to
	  << "\n";
    }
  }
} // end assignMessageTags()

void
DetailedTasks::add(DetailedTask* task)
{
  tasks.push_back(task);
}

#if 0
vector<DetailedReq*>&
DetailedTasks::getInitialRequires()
{
#if 0
  for(DetailedReq* req = task->getRequires();
      req != 0; req = req->next){
    if(req->req->dw == Task::OldDW)
      initreqs.push_back(req);
  }
#else
  if( mixedDebug.active() ) {
    cerrLock.lock();
    NOT_FINISHED("DetailedTasks::add");
    cerrLock.unlock();
  }
#endif
  cerr << initreqs.size() << " initreqs\n";
  return initreqs;
}
#endif

void
DetailedTasks::computeLocalTasks(int me)
{
  initiallyReadyTasks_ = TaskQueue();
  if(localtasks.size() != 0)
    return;
  for(int i=0;i<(int)tasks.size();i++){
    DetailedTask* task = tasks[i];

    if(task->getAssignedResourceIndex() == me
       || task->getTask()->getType() == Task::Reduction) {
      localtasks.push_back(task);

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
    ASSERT(is_sorted(patches->getVector().begin(), patches->getVector().end(),
		     Patch::Compare()));    
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
DetailedTask::doit(const ProcessorGroup* pg, DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
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
  
  task->doit(pg, patches, matls, old_dw, new_dw);
}

void
DetailedTask::scrub(DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  Handle<PatchSubset> default_patches = scinew PatchSubset();
  Handle<MaterialSubset> default_matls = scinew MaterialSubset();
  default_patches->add(0);
  default_matls->add(-1);

  list<VarLabelMatlPatch>::iterator vmpIter;    
  
  if (taskGroup->doScrubNew()) {
    // handle scrub counters -- scrubbing when the count gets to zero
    for (Task::Dependency* dep = task->getComputes(); dep != 0;
	 dep = dep->next){
      constHandle<PatchSubset> compPatches =
	dep->getPatchesUnderDomain(patches);
      constHandle<MaterialSubset> compMatls =
	dep->getMaterialsUnderDomain(matls);
      if (dep->var->typeDescription() &&
	  dep->var->typeDescription()->isReductionVariable()) {
	compPatches = default_patches.get_rep();
      }
      else if (compPatches == 0) {
	compPatches = default_patches.get_rep();
      }
      if (compMatls == 0) {
	compMatls = default_matls.get_rep();
      }
      for (int m = 0; m < compMatls->size(); m++) {
	int matl = compMatls->get(m);
	for (int p = 0; p < compPatches->size(); p++) {
	  const Patch* patch = compPatches->get(p);
	  int scrubCount = taskGroup->getScrubCount(dep->var, matl, patch);
	  if (scrubCount != NO_SCRUB) {
	    new_dw->setScrubCountIfZero(dep->var, matl, patch, scrubCount);
	  }
	}
      }
    }
    for (vmpIter = requiresForNewData_.begin();
	 vmpIter != requiresForNewData_.end(); ++vmpIter) {
      const VarLabelMatlPatch& vmp = *vmpIter;
      int scrubCount = taskGroup->getScrubCount(vmp.label_, vmp.matlIndex_,
						vmp.patch_);
      if (scrubCount != NO_SCRUB) {
	new_dw->decrementScrubCount(vmp.label_, vmp.matlIndex_,
				    vmp.patch_, 1, scrubCount);
      }
    }
  }
  for (vmpIter = requiresForOldData_.begin();
       vmpIter != requiresForOldData_.end(); ++vmpIter) {
    const VarLabelMatlPatch& vmp = *vmpIter;    
    int scrubCount = taskGroup->getOldDWScrubCount(vmp.label_, vmp.matlIndex_,
						   vmp.patch_);
    old_dw->decrementScrubCount(vmp.label_, vmp.matlIndex_,
				vmp.patch_, 1, scrubCount);
  }

  if (task->getType() == Task::InitialSend) {
    if (taskGroup->scrubExtraneousOldDW_) {
      taskGroup->actuallyScrubExtraneous(old_dw);
    }
  }
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << this << " DetailedTask::done doit() for task " 
	       << task << "\n";
    cerrLock.unlock();
  }
}

void DetailedTasks::scrubExtraneousOldDW()
{
  // scrub after doing the initial send just in case anything something is
  // required by another processor and not this one.
  scrubExtraneousOldDW_ = true;
}

void DetailedTasks::actuallyScrubExtraneous(DataWarehouse* old_dw)
{
  // must add scrub count stuff here so only the non-required data will
  // get scrubbed
  for (ScrubCountMap::iterator iter = oldDWScrubCountMap_.begin();
       iter != oldDWScrubCountMap_.end(); ++iter) {
    VarLabelMatlPatch vmp = iter->first;
    // If it doesn't exist, it will be received later, so don't worry about it.
    if (old_dw->exists(vmp.label_, vmp.matlIndex_, vmp.patch_)) {
	old_dw->setScrubCountIfZero(vmp.label_, vmp.matlIndex_, vmp.patch_,
				    iter->second);
    }
  }
  
  old_dw->scrubExtraneous();
  scrubExtraneousOldDW_ = false;
}

void
DetailedTasks::scrubCountDependency(DetailedTask* to,
				    Task::Dependency* req,
				    const Patch *fromPatch,
				    int matl, Task::WhichDW dw)
{
  // Dav's conjectures on how this works:
  //   This function is called once for each time a dependency is found
  //   while creating the task graph.  After it has been called X times,
  //   scrubCountMap_[VAR] == X.  Later, each time a task that uses VAR,
  //   scrubCountMap_[VAR] is decremented.  When it reaches 0, the VAR
  //   can be scrubbed as no one will be using it after that.
  //   
  // Questions:
  //   What is "doScrubNew()" used for?
  //

  if (matl < 0) fromPatch = 0;
  if (dw == Task::NewDW) {
    if (doScrubNew()) {
      int & scrubCount =
	scrubCountMap_[VarLabelMatlPatch(req->var, matl, fromPatch)];
      if (scrubCount != NO_SCRUB) {
	scrubCount++;    
	to->addRequiresForNewData(req->var, matl, fromPatch);
      }
    }
  }
  else {
    oldDWScrubCountMap_[VarLabelMatlPatch(req->var, matl, fromPatch)]++;
    to->addRequiresForOldData(req->var, matl, fromPatch);
    // Don't scrub what may be required on the next timestep
    scrubCountMap_[VarLabelMatlPatch(req->var, matl, fromPatch)] = NO_SCRUB;
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


void
DetailedTasks::possiblyCreateDependency(DetailedTask* from,
					Task::Dependency* comp,
					const Patch* fromPatch,
					DetailedTask* to,
					Task::Dependency* req,
					const Patch */*toPatch*/,
					int matl,
					const IntVector& low,
					const IntVector& high)
{
  // TODO - maybe still create internal depencies for threaded scheduler?
  // TODO - perhaps move at least some of this to TaskGraph?
  ASSERT(from->getAssignedResourceIndex() != -1);
  ASSERT(to->getAssignedResourceIndex() != -1);
  if(dbg.active()) {
    dbg << *to << " depends on " << *from << "\n";
    if(comp)
      dbg << "From comp " << *comp;
    else
      dbg << "From OldDW ";
    dbg << " to req " << *req << '\n';
  }

  if(from->getAssignedResourceIndex() == to->getAssignedResourceIndex()) {
    to->addInternalDependency(from, req->var);
    return;
  }
  int toresource = to->getAssignedResourceIndex();
  DependencyBatch* batch = from->getComputes();
  for(;batch != 0; batch = batch->comp_next){
    if(batch->to == toresource)
      break;
  }
  if(!batch){
    batch = scinew DependencyBatch(toresource, from, to);
    batches.push_back(batch);
    from->addComputes(batch);
    bool newRequireBatch = to->addRequires(batch);
    ASSERT(newRequireBatch);
    if(dbg.active())
      dbg << "NEW BATCH!\n";
  }
  else if (mustConsiderInternalDependencies_) { // i.e. threaded mode
    if (to->addRequires(batch)) {
      // this is a new requires batch for this task, so add
      // to the batch's toTasks.
      batch->toTasks.push_back(to);
    }
    if(dbg.active())
      dbg << "USING PREVIOUSLY CREATED BATCH!\n";
  }
  DetailedDep* dep = batch->head;
  for(;dep != 0; dep = dep->next){
    if(fromPatch == dep->fromPatch && matl == dep->matl
       && (req == dep->req
	   || (req->var->equals(dep->req->var)
	       && req->dw == dep->req->dw)))
      break;
  }
  if(!dep){
    dep = scinew DetailedDep(batch->head, comp, req, to, fromPatch, matl, 
			     low, high);
    batch->head = dep;
    if(dbg.active()) {
      dbg << "ADDED " << low << " " << high << ", fromPatch = ";
      if (fromPatch)
	dbg << fromPatch->getID() << '\n';
      else
	dbg << "NULL\n";	
    }
  } else {
    dep->toTasks.push_back(to);
    IntVector l = Min(low, dep->low);
    IntVector h = Max(high, dep->high);
    IntVector d1 = h-l;
    IntVector d2 = high-low;
    IntVector d3 = dep->high-dep->low;
    int v1 = d1.x()*d1.y()*d1.z();
    int v2 = d2.x()*d2.y()*d2.z();
    int v3 = d3.x()*d3.y()*d3.z();
    if(v1 > v2+v3){
      // If we get this, perhaps we should allow multiple deps so
      // that we do not communicate more of the patch than necessary
      static int warned=false;
      if(!warned){
	cerr << "WARNING: Possible extra communication between patches!\n";
	cerr << "This warning will only appear once\n";
	
	warned=true;
      }
    }
    if(dbg.active()){
      dbg << "EXTENDED from " << dep->low << " " << dep->high << " to " << l << " " << h << "\n";
      dbg << *req->var << '\n';
      dbg << *dep->req->var << '\n';
      if(comp)
	dbg << *comp->var << '\n';
      if(dep->comp)
	dbg << *dep->comp->var << '\n';
    }
    dep->low=l;
    dep->high=h;
  }
}

DetailedTask*
DetailedTasks::getOldDWSendTask(int proc)
{
  // These are the first N tasks
  return tasks[proc];
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
    }
    else {
      foundIt->second->addVarLabel(var);
    }
  }
}

void
DetailedTask::done(DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(old_dw, new_dw);  
  
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
    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << cnt << ": " << *(dep->dependentTask->task) << "\n";
      cerrLock.unlock();
    }
    dep->dependentTask->dependencySatisfied(dep);
    cnt++;
  }
}

void DetailedTask::dependencySatisfied(InternalDependency* dep)
{
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Depend satisfied for " << *(dep->dependentTask->getTask()) 
	       << "\n";
    cerrLock.unlock();
  }

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
  //DetailedTask* nextTask = readyTasks_.top();
  readyTasks_.pop();
#if !defined( _AIX )
  readyQueueMutex_.unlock();
#endif
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

void DetailedTasks::initializeBatches()
{
//  cerr << "Initializing Batches\n";
  for (int i = 0; i < (int)batches.size(); i++) {
    batches[i]->reset();
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

/*
bool DependencyBatch::makeMPIRequest()
{
  if (toTasks.size() > 1) {
    if (!received_) {
      lock_->lock();
      if (!received_) {
	if (!madeMPIRequest_) {
	  madeMPIRequest_ = true;
	  lock_->unlock();
	  return true;
	}
	else {
	  lock_->unlock();
	  return false;
	}
      }
      lock_->unlock();
    }
    return false;
  }
  else {
    return true;
  }
}

bool DependencyBatch::waitForMPIRequest()
{
  if (toTasks.size() > 1) {
    if (!received_) {
      lock_->lock();
      if (!received_) {
	if (!madeMPIRequest_) {
	  madeMPIRequest_ = true;
	  lock_->unlock();
	  return true;
	}
	else {
	  cv_->wait(*lock_);
	  lock_->unlock();
	  return false;
	}
      }
      lock_->unlock();
    }
    return false;
  }
  else
    return true;
}
*/

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
  //cv_->conditionBroadcast(); -- replaced with mpi "listeners" below
  //lock_->unlock();
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Received batch message " << messageTag 
	       << " from task " << *fromTask << "\n";
    
    for (DetailedDep* dep = head; dep != 0; dep = dep->next)
      mixedDebug << "\tSatisfying " << *dep << "\n";
    cerrLock.unlock();
  }

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
}

void DetailedTasks::logMemoryUse(ostream& out, unsigned long& total,
				 const std::string& tag)
{
  ostringstream elems1;
  elems1 << tasks.size();
  logMemory(out, total, tag, "tasks", "DetailedTask", 0, -1,
	    elems1.str(), tasks.size()*sizeof(DetailedTask), 0);
  ostringstream elems2;
  elems2 << batches.size();
  logMemory(out, total, tag, "batches", "DependencyBatch", 0, -1,
	    elems2.str(), batches.size()*sizeof(DependencyBatch), 0);
  int ndeps=0;
  for(int i=0;i<(int)batches.size();i++){
    for(DetailedDep* p=batches[i]->head; p != 0; p = p->next)
      ndeps++;
  }
  ostringstream elems3;
  elems3 << ndeps;
  logMemory(out, total, tag, "deps", "DetailedDep", 0, -1,
	    elems3.str(), ndeps*sizeof(DetailedDep), 0);
}

void DetailedTasks::emitEdges(DOM_Element edgesElement, int rank)
{
  for (int i = 0; i < (int)tasks.size(); i++) {
    if (tasks[i]->getAssignedResourceIndex() == rank) {
      tasks[i]->emitEdges(edgesElement);
    }
  }
}

void DetailedTask::emitEdges(DOM_Element edgesElement)
{
  map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = reqs.begin(); req_iter != reqs.end(); req_iter++) {
    DetailedTask* fromTask = (*req_iter).first->fromTask;
    DOM_Element edge = edgesElement.getOwnerDocument().createElement("edge");
    appendElement(edge, "source", fromTask->getName());
    appendElement(edge, "target", getName());
    edgesElement.appendChild(edge);
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
    DOM_Element edge = edgesElement.getOwnerDocument().createElement("edge");
    appendElement(edge, "source", fromTask->getName());
    appendElement(edge, "target", getName());
    edgesElement.appendChild(edge);
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
