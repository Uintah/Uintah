#include <Packages/Uintah/CCA/Components/Scheduler3/DetailedTasks3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/TaskGraph3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/Scheduler3Common.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time) 
// From: sus.cc
extern SCIRun::Mutex cerrLock;
extern DebugStream brydbg;
static DebugStream dbg("TaskGraph", false);
static DebugStream scrubout("Scrubbing", false);

DetailedTasks3::DetailedTasks3(Scheduler3Common* sc, const ProcessorGroup* pg,
			     const TaskGraph3* taskgraph,
			     bool mustConsiderInternalDependencies /*= false*/)
  : sc_(sc), d_myworld(pg), taskgraph_(taskgraph),
    mustConsiderInternalDependencies_(mustConsiderInternalDependencies),
    currentDependencyGeneration_(1),
    readyQueueMutex_("DetailedTasks3 Ready Queue"),
    readyQueueSemaphore_("Number of Ready DetailedTasks3", 0)
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
    tasks_[i]=scinew DetailedTask3(stasks_[i], 0, 0, this);
    tasks_[i]->assignResource(i);
  }
}

DetailedTasks3::~DetailedTasks3()
{
  //  for(int i=0;i<(int)batches_.size();i++)
  //  delete batches_[i];

  for(int i=0;i<(int)tasks_.size();i++)
    delete tasks_[i];

  for(int i=0;i<(int)stasks_.size();i++)
    delete stasks_[i];
}

void
DetailedTasks3::assignMessageTags(int me)
{
} // end assignMessageTags()

void
DetailedTasks3::add(DetailedTask3* task)
{
  tasks_.push_back(task);
}


void
DetailedTasks3::computeLocalTasks(int me)
{
  initiallyReadyTasks_ = TaskQueue();
  if(localtasks_.size() != 0)
    return;
  for(int i=0;i<(int)tasks_.size();i++){
    DetailedTask3* task = tasks_[i];

    ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
    if(task->getAssignedResourceIndex() == me
       || task->getTask()->getType() == Task::Reduction) {
      localtasks_.push_back(task);

      if (task->areInternalDependenciesSatisfied()) {
	initiallyReadyTasks_.push(task);

	if( dbg.active() ) {
	  cerrLock.lock();
	  dbg << "Initially Ready Task: " 
		     << task->getTask()->getName() << "\n";
	  cerrLock.unlock();
	}
      }
    }
  }
}

DetailedTask3::DetailedTask3(Task* task, const PatchSubset* patches,
			   const MaterialSubset* matls, DetailedTasks3* taskGroup)
  : task(task), patches(patches), matls(matls), // comp_head(0),
    taskGroup(taskGroup),
    //    numPendingInternalDependencies(0),
    //    internalDependencyLock("DetailedTask3 Internal Dependencies"),
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

DetailedTask3::~DetailedTask3()
{
  if(patches && patches->removeReference())
    delete patches;
  if(matls && matls->removeReference())
    delete matls;
}

void
DetailedTask3::doit(const ProcessorGroup* pg,
		   vector<PatchBasedDataWarehouse3P>& oddws,
		   vector<DataWarehouseP>& dws)
{
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

void DetailedTasks3::initializeScrubs(vector<PatchBasedDataWarehouse3P>& dws)
{
  if(scrubout.active())
    scrubout << "Begin initialize scrubs\n";
  for(int i=0;i<(int)dws.size();i++){
    if(dws[i] != 0 && dws[i]->getScrubMode() == DataWarehouse::ScrubComplete){
      scrubout << "Initializing scrubs on dw: " << dws[i]->getID() << '\n';
      dws[i]->initializeScrubs(i, scrubCountMap_);
    }
  }
  if(scrubout.active())
    scrubout << "End initialize scrubs\n";
}

void
DetailedTask3::scrub(vector<PatchBasedDataWarehouse3P>& dws)
{
  const Task* task = getTask();

  if(scrubout.active())
    scrubout << "Starting scrub after task: " << *this << '\n';
  const set<const VarLabel*, VarLabel::Compare>& initialRequires
    = taskGroup->getTaskGraph()->getInitialRequiredVars();
  // Decrement the scrub count for each of the required variables
  for(const Task::Dependency* req = task->getRequires();
      req != 0; req=req->next){
    if(req->var->typeDescription()->getType() != TypeDescription::ReductionVariable){
      int dw = req->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if(scrubmode == DataWarehouse::ScrubComplete ||
	 (scrubmode == DataWarehouse::ScrubNonPermanent &&
	  initialRequires.find(req->var) == initialRequires.end())){
	constHandle<PatchSubset> patches = req->getPatchesUnderDomain(getPatches());
	constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(getMaterials());
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  Patch::selectType neighbors;
	  IntVector low, high;
	  patch->computeVariableExtents(req->var->typeDescription()->getType(),
					req->var->getBoundaryLayer(),
					req->gtype, req->numGhostCells,
					neighbors, low, high);
	  for(int i=0;i<neighbors.size();i++){
	    const Patch* neighbor=neighbors[i];
	    for (int m=0;m<matls->size();m++){
	      if(scrubout.active()){
		scrubout << "  decrementing scrub count for requires of " << dws[dw]->getID() << "/" << neighbor->getID() << "/" << matls->get(m) << "/" << req->var->getName() << '\n';
	      }
	      dws[dw]->decrementScrubCount(req->var, matls->get(m), neighbor);
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
      constHandle<PatchSubset> patches = mod->getPatchesUnderDomain(getPatches());
      constHandle<MaterialSubset> matls = mod->getMaterialsUnderDomain(getMaterials());
      if(mod->var->typeDescription()->getType() != TypeDescription::ReductionVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++){
	    if(scrubout.active())
	      scrubout << "  decrementing scrub count for modifies of " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << mod->var->getName() << '\n';
	    dws[dw]->decrementScrubCount(mod->var, matls->get(m), patch);
	  }
	}
      }
    }
  }
  
  // Set the scrub count for each of the computes variables
  for(const Task::Dependency* comp = task->getComputes();
      comp != 0; comp=comp->next){
    if(comp->var->typeDescription()->getType() != TypeDescription::ReductionVariable){
      int dw = comp->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if(scrubmode == DataWarehouse::ScrubComplete ||
	 (scrubmode == DataWarehouse::ScrubNonPermanent &&
	  initialRequires.find(comp->var) == initialRequires.end())){
	constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(getPatches());
	constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(getMaterials());
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++){
	    int matl = matls->get(m);
	    int count;
	    if(taskGroup->getScrubCount(comp->var, matl, patch, dw, count)){
	      if(scrubout.active())
		scrubout << "  setting scrub count for computes of " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << " to " << count << '\n';
	      dws[dw]->setScrubCount(comp->var, matl, patch, count);
	    } else {
	      // Not in the scrub map, must be never needed...
	      if(scrubout.active())
		scrubout << "  trashing variable immediately after compute: " << dws[dw]->getID() << "/" << patch->getID() << "/" << matls->get(m) << "/" << comp->var->getName() << '\n';
	      dws[dw]->scrub(comp->var, matl, patch);
	    }
	  }
	}
      }
    }
  }
}

void DetailedTasks3::addScrubCount(const VarLabel* var, int matlindex,
				  const Patch* patch, int dw)
{
  if(patch->isVirtual())
    patch = patch->getRealPatch();
  VarLabelMatlDW<Patch> key(var, matlindex, patch, dw);
  ScrubCountMap::iterator iter = scrubCountMap_.find(key);
  if(iter == scrubCountMap_.end()){
    scrubCountMap_.insert(make_pair(key, 1));
  } else {
    iter->second++;
  }
}

void DetailedTasks3::setScrubCount(const VarLabel* label, int matlIndex,
				  const Patch* patch, int dw,
				  vector<PatchBasedDataWarehouse3P>& dws)
{
  ASSERT(!patch->isVirtual());
  DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
  const set<const VarLabel*, VarLabel::Compare>& initialRequires
    = getTaskGraph()->getInitialRequiredVars();
  if(scrubmode == DataWarehouse::ScrubComplete ||
     (scrubmode == DataWarehouse::ScrubNonPermanent &&
      initialRequires.find(label) == initialRequires.end())){
    int scrubcount;
    if(!getScrubCount(label, matlIndex, patch, dw, scrubcount)){
      SCI_THROW(InternalError("No scrub count for received MPIVariable: "+label->getName()));
    }
    if(scrubout.active())
      scrubout << "setting scrubcount for recv of " << dw << "/" << patch->getID() << "/" << matlIndex << "/" << label->getName() << ": " << scrubcount << '\n';
    dws[dw]->setScrubCount(label, matlIndex, patch, scrubcount);
  }
}

bool DetailedTasks3::getScrubCount(const VarLabel* label, int matlIndex,
				  const Patch* patch, int dw, int& count)
{
  ASSERT(!patch->isVirtual());
  VarLabelMatlDW<Patch> key(label, matlIndex, patch, dw);
  ScrubCountMap::iterator iter = scrubCountMap_.find(key);

  if(iter == scrubCountMap_.end())
    return false;
  count=iter->second;
  return true;
}

void DetailedTasks3::createScrubCounts()
{
  scrubCountMap_.clear();
  
  // Go through each of the tasks and determine which variables it will require
  for(int i=0;i<(int)localtasks_.size();i++){
    DetailedTask3* dtask = localtasks_[i];
    const Task* task = dtask->getTask();
    for(const Task::Dependency* req = task->getRequires(); req != 0; req=req->next){
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int dw = req->mapDataWarehouse();
      if(req->var->typeDescription()->getType() != TypeDescription::ReductionVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  Patch::selectType neighbors;
	  IntVector low, high;
	  patch->computeVariableExtents(req->var->typeDescription()->getType(),
					req->var->getBoundaryLayer(),
					req->gtype, req->numGhostCells,
					neighbors, low, high);
	  for(int i=0;i<neighbors.size();i++){
	    const Patch* neighbor=neighbors[i];
	    for (int m=0;m<matls->size();m++)
	      addScrubCount(req->var, matls->get(m), neighbor, dw);
	  }
	}
      }
    }
    for(const Task::Dependency* req = task->getModifies(); req != 0; req=req->next){
      constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());
      int dw = req->mapDataWarehouse();
      if(req->var->typeDescription()->getType() != TypeDescription::ReductionVariable){
	for(int i=0;i<patches->size();i++){
	  const Patch* patch = patches->get(i);
	  for (int m=0;m<matls->size();m++)
	    addScrubCount(req->var, matls->get(m), patch, dw);
	}
      }
    }
  }
  if(scrubout.active()){
    scrubout << "scrub counts:\n";
    scrubout << "DW/Patch/Matl/Label\tCount\n";
    for(ScrubCountMap::iterator iter = scrubCountMap_.begin();
	iter != scrubCountMap_.end(); iter++){
      const VarLabelMatlDW<Patch>& rec = iter->first;
      scrubout << rec.dw_ << '/' << (rec.domain_?rec.domain_->getID():0) << '/'
	       << rec.matlIndex_ << '/' <<  rec.label_->getName()
	       << "\t\t" << iter->second << '\n';
    }
    scrubout << "end scrub counts\n";
  }
}

void DetailedTask3::findRequiringTasks(const VarLabel* var,
				      list<DetailedTask3*>& requiringTasks)
{
  // find requiring tasks
  /*
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
  map<DetailedTask3*, InternalDependency*>::iterator internalDepIter;
  for (internalDepIter = internalDependents.begin();
       internalDepIter != internalDependents.end(); ++internalDepIter) {
    if (internalDepIter->second->vars.find(var) !=
	internalDepIter->second->vars.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
  */
}


void
DetailedTasks3::possiblyCreateDependency(DetailedTask3* from,
					Task::Dependency* comp,
					const Patch* fromPatch,
					DetailedTask3* to,
					Task::Dependency* req,
					const Patch */*toPatch*/,
					int matl,
					const IntVector& low,
					const IntVector& high)
{
}

DetailedTask3*
DetailedTasks3::getOldDWSendTask(int proc)
{
  // These are the first N tasks
  return tasks_[proc];
}

/*
void
DetailedTask3::addComputes(DependencyBatch* comp)
{
  comp->comp_next=comp_head;
  comp_head=comp;
}

bool
DetailedTask3::addRequires(DependencyBatch* req)
{
  // return true if it is adding a new batch
  return reqs.insert(make_pair(req, req)).second;
}
*/
void
DetailedTask3::addInternalDependency(DetailedTask3* prerequisiteTask,
				    const VarLabel* var)
{
  /*
  if (taskGroup->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    map<DetailedTask3*, InternalDependency*>::iterator foundIt =
      prerequisiteTask->internalDependents.find(this);
    if (foundIt == prerequisiteTask->internalDependents.end()) {
      internalDependencies.push_back(InternalDependency(prerequisiteTask,
							this, var,
							0)); // not satisfied
      prerequisiteTask->
	internalDependents[this] = &internalDependencies.back();
      numPendingInternalDependencies = internalDependencies.size();
    }
    else {
      foundIt->second->addVarLabel(var);
    }
  }
*/
}

void
DetailedTask3::done(vector<PatchBasedDataWarehouse3P>& dws)
{
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);
  
  /*
  if( dbg.active() ) {
    cerrLock.lock();
    dbg << "This: " << this << " is done with task: " << task << "\n";
    dbg << "Name is: " << task->getName()
      	       << " which has (" << internalDependents.size() 
      	       << ") tasks waiting on it:\n";
    cerrLock.unlock();
  }

  int cnt = 1000;
  map<DetailedTask3*, InternalDependency*>::iterator iter;
  for (iter = internalDependents.begin(); iter != internalDependents.end(); 
       iter++) {
    InternalDependency* dep = (*iter).second;
    if( dbg.active() ) {
      cerrLock.lock();
      dbg << cnt << ": " << *(dep->dependentTask->task) << "\n";
      cerrLock.unlock();
    }
    dep->dependentTask->dependencySatisfied(dep);
    cnt++;
  }
  */
}

ostream&
operator<<(ostream& out, const DetailedTask3& task)
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

#if 0
ostream&
operator<<(ostream& out, const DetailedDep& dep)
{
  /*
  out << dep.req->var->getName();
  if (dep.isNonDataDependency())
    out << " non-data dependency";
  else
    out << " on patch " << dep.fromPatch->getID();
  out << ", matl " << dep.matl << ", low=" << dep.low << ", high=" << dep.high;
  */
  return out;
}
#endif

void
DetailedTasks3::internalDependenciesSatisfied(DetailedTask3* task)
{
  if( dbg.active() ) {
    cerrLock.lock();
    dbg << "Begin internalDependenciesSatisfied\n";
    cerrLock.unlock();
  }
#if !defined( _AIX )
  readyQueueMutex_.lock();
#endif

  readyTasks_.push(task);

  if( dbg.active() ) {
    cerrLock.lock();
    dbg << *task << " satisfied.  Now " 
	       << readyTasks_.size() << " ready.\n";
    cerrLock.unlock();
  }
#if !defined( _AIX )
  // need to make a non-binary semaphore under aix for this to work.
  readyQueueSemaphore_.up();
  readyQueueMutex_.unlock();
#endif
}

DetailedTask3*
DetailedTasks3::getNextInternalReadyTask()
{
  // Block until the list has an item in it.
#if !defined( _AIX )
  readyQueueSemaphore_.down();
  readyQueueMutex_.lock();
#endif
  DetailedTask3* nextTask = readyTasks_.front();
  //DetailedTask3* nextTask = readyTasks_.top();
  readyTasks_.pop();
#if !defined( _AIX )
  readyQueueMutex_.unlock();
#endif
  return nextTask;
}

void
DetailedTasks3::initTimestep()
{
  readyTasks_ = initiallyReadyTasks_;
#if !defined( _AIX )
  readyQueueSemaphore_.up((int)readyTasks_.size());
#endif
  incrementDependencyGeneration();
  initializeBatches();
}

void DetailedTasks3::incrementDependencyGeneration()
{
  if (currentDependencyGeneration_ >= ULONG_MAX)
    SCI_THROW(InternalError("DetailedTasks3::currentDependencySatisfyingGeneration has overflowed"));
  currentDependencyGeneration_++;
}

void DetailedTasks3::initializeBatches()
{
  /*
  for (int i = 0; i < (int)batches_.size(); i++) {
    batches_[i]->reset();
  }
  */
}

/*
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
  
  if( dbg.active() ) {
    cerrLock.lock();
    dbg << "Received batch message " << messageTag 
	       << " from task " << *fromTask << "\n";
    
    for (DetailedDep* dep = head; dep != 0; dep = dep->next)
      dbg << "\tSatisfying " << *dep << "\n";
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
*/
void DetailedTasks3::logMemoryUse(ostream& out, unsigned long& total,
				 const std::string& tag)
{
  ostringstream elems1;
  elems1 << tasks_.size();
  logMemory(out, total, tag, "tasks", "DetailedTask3", 0, -1,
	    elems1.str(), tasks_.size()*sizeof(DetailedTask3), 0);
  ostringstream elems2;

  /*
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
  */
}

void DetailedTasks3::emitEdges(ProblemSpecP edgesElement, int rank)
{
  for (int i = 0; i < (int)tasks_.size(); i++) {
    ASSERTRANGE(tasks_[i]->getAssignedResourceIndex(), 0, d_myworld->size());
    if (tasks_[i]->getAssignedResourceIndex() == rank) {
      tasks_[i]->emitEdges(edgesElement);
    }
  }
}

void DetailedTask3::emitEdges(ProblemSpecP edgesElement)
{
  /*
  map<DependencyBatch*, DependencyBatch*>::iterator req_iter;
  for (req_iter = reqs.begin(); req_iter != reqs.end(); req_iter++) {
    DetailedTask3* fromTask = (*req_iter).first->fromTask;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  list<InternalDependency>::iterator iter;
  for (iter = internalDependencies.begin();
       iter != internalDependencies.end(); iter++) {
    DetailedTask3* fromTask = (*iter).prerequisiteTask;
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
  */
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

string DetailedTask3::getName() const
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
