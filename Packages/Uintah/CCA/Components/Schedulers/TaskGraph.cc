

#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/NotFinished.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <multimap.h>
#include <sstream>
#include <unistd.h>

using namespace Uintah;

using namespace SCIRun;
using std::cerr;

static DebugStream dbg("TaskGraph", false);

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

#define DAV_DEBUG 0

TaskGraph::TaskGraph()
{
}

TaskGraph::~TaskGraph()
{
  initialize(); // Frees all of the memory...
}

void
TaskGraph::initialize()
{
  for(vector<Task*>::iterator iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
    delete *iter;

  for(vector<Task::Edge*>::iterator iter = edges.begin(); iter != edges.end(); iter++)
    delete *iter;

  d_tasks.clear();
  d_initRequires.clear();
  d_initRequiredVars.clear();
  edges.clear();
}

template<class T>
bool csoverlaps(const ComputeSubset<T>* s1, const ComputeSubset<T>* s2)
{
  if(s1 == s2)
    return true;
  if(s1->size() == 0 || s2->size() == 0)
    return false;
#if 0
  T el1 = s1->get(0);
  for(int i=1;i<s1->size();i++){
    T el = s1->get(i);
    if(el <= el1){
      cerr << "Set not sorted: " << el1 << ", " << el << '\n';
    }
    el1=el;
  }
  T el2 = s2->get(0);
  for(int i=1;i<s2->size();i++){
    T el = s2->get(i);
    if(el <= el2){
      cerr << "Set not sorted: " << el2 << ", " << el << '\n';
    }
    el2=el;
  }
#endif
  int i1=0;
  int i2=0;
  for(;;){
    if(s1->get(i1) == s2->get(i2)){
      return true;
    } else if(s1->get(i1) < s2->get(i2)){
      if(++i1 == s1->size())
	break;
    } else {
      if(++i2 == s2->size())
	break;
    }
  }
  return false;
}

bool
TaskGraph::overlaps(Task::Dependency* comp, Task::Dependency* req) const
{
  const PatchSubset* ps1 = comp->patches;
  if(!ps1){
    if(!comp->task->getPatchSet())
      return false;
    ps1 = comp->task->getPatchSet()->getUnion();
  }

  const PatchSubset* ps2 = req->patches;
  if(!ps2){
    if(!req->task->getPatchSet())
      return false;
    ps2 = req->task->getPatchSet()->getUnion();
  }
  if(!csoverlaps(ps1, ps2))
    return false;

  const MaterialSubset* ms1 = comp->matls;
  if(!ms1){
    if(!comp->task->getMaterialSet())
      return false;
    ms1 = comp->task->getMaterialSet()->getUnion();
  }
  const MaterialSubset* ms2 = req->matls;
  if(!ms2){
    if(!req->task->getMaterialSet())
      return false;
    ms2 = req->task->getMaterialSet()->getUnion();
  }
  if(!csoverlaps(ms1, ms2))
    return false;
  return true;
}

/* overloaded addRequires used in setupTaskConnections */

inline void addRequires(Task* task, Task::WhichDW dw, const VarLabel* var,
			const PatchSet* ps, const MaterialSet* ms)
{  
  for(int p=0;p<ps->size();p++){
    const PatchSubset* pss = ps->getSubset(p);
    for(int m=0;m<ms->size();m++){
      const MaterialSubset* mss = ms->getSubset(m);
      task->requires(dw, var, pss, mss, Ghost::None);
    }
  }
}

inline void addRequires(Task* task, Task::WhichDW dw, const VarLabel* var,
			const PatchSubset* pss, const MaterialSet* ms)
{  
  for(int m=0;m<ms->size();m++){
    const MaterialSubset* mss = ms->getSubset(m);
    task->requires(dw, var, pss, mss, Ghost::None);
  }
}
 
inline void addRequires(Task* task, Task::WhichDW dw, const VarLabel* var,
			const PatchSet* ps, const MaterialSubset* mss)
{  
  for(int p=0;p<ps->size();p++){
    const PatchSubset* pss = ps->getSubset(p);
    task->requires(dw, var, pss, mss, Ghost::None);
  }
}

// setupTaskConnections also adds Reduction Tasks to the graph...
void
TaskGraph::setupTaskConnections()
{
  vector<Task*>::iterator iter;

  // Look for all of the reduction variables - we must treat those
  // special.  Create a fake task that performs the reduction
  // While we are at it, ensure that we aren't producing anything
  // into a frozen data warehouse
  typedef map<const VarLabel*, Task*, VarLabel::Compare> ReductionTasksMap;
  ReductionTasksMap reductionTasks;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if (task->isReductionTask())
      continue; // already a reduction task so skip it

    for(Task::Dependency* comp = task->getComputes();
	comp != 0; comp=comp->next){
      if(comp->dw == Task::OldDW){
	throw InternalError("Variable produced in old datawarehouse: "
			    +comp->var->getName());
      } else if(comp->var->typeDescription()->isReductionVariable()){
	// Look up this variable in the reductionTasks map
	const VarLabel* var = comp->var;
	const PatchSet* ps = task->getPatchSet();
	const MaterialSet* ms = task->getMaterialSet();

	ReductionTasksMap::iterator it=reductionTasks.find(var);
	if(it == reductionTasks.end()){
	  // No reduction task yet, create one
	  Task* newtask = scinew Task(var->getName()+" reduction",
				      Task::Reduction);
	  // compute for all patches but some set of materials
	  // (maybe global material, but not necessarily)
	  if (comp->matls != 0)
	    newtask->computes(var, comp->matls, Task::OutOfDomain);
	  else {
	    for(int m=0;m<ms->size();m++)
	      newtask->computes(var, ms->getSubset(m), Task::OutOfDomain);
	  }
	  reductionTasks[var]=newtask;
	  it = reductionTasks.find(var);
	}

	// Make the reduction task require its variable for the appropriate
	// patch and materials subset(s).
	if (comp->patches != 0) {
	  if (comp->matls != 0)
	    it->second->requires(comp->dw, var, comp->patches, comp->matls,
				 Ghost::None);
	  else
	    addRequires(it->second, comp->dw, var, comp->patches, ms);
	}
	else if (comp->matls != 0)
	  addRequires(it->second, comp->dw, var, ps, comp->matls);
	else
	  addRequires(it->second, comp->dw, var, ps, ms);
      }
    }
  }

  // Add the new reduction tasks to the list of tasks
  for(ReductionTasksMap::iterator it = reductionTasks.begin();
      it != reductionTasks.end(); it++){
    addTask(it->second, 0, 0);
  }

  // Gather the comps for the tasks into a map
  CompMap comps;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    for(Task::Dependency* comp = task->getComputes();
	comp != 0; comp=comp->next){
      comps.insert(make_pair(comp->var, comp));
    }
  }

#if 0
  // Make sure that two tasks do not compute the same result
  const Task::compType& comps = task->getComputes();
  for(Task::compType::const_iterator dep = comps.begin();
      dep != comps.end(); dep++){
    TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
    if (!dep->d_var->allowsMultipleComputes()) {
      actype::iterator aciter = d_allcomps.find(p);
      if(aciter != d_allcomps.end()){
	cerr << "First task:\n";
	task->displayAll(cerr);
	cerr << "Second task:\n";
	aciter->second->d_task->displayAll(cerr);
	throw InternalError("Two tasks compute the same result: "+dep->d_var->getName()+" (tasks: "+task->getName()+" and "+aciter->second->d_task->getName()+")");
      }
    }
    d_allcomps.insert(actype::value_type(p, dep));
  }
#else
  cerrLock.lock();
  NOT_FINISHED("new task stuff (addTask - skip)");
  cerrLock.unlock();
#endif

#if 0 // moved to addTask
  // Gather a list of requirements
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    for(Task::Dependency* req = task->getRequires();
	req != 0; req=req->next){
      if(req->dw == Task::OldDW) {
	d_initRequires.push_back(req);
	d_initRequiredVars.insert(req->var);
      }
    }
    // Set these to false to prepare for the next loop.
    task->visited = false;
  }
#endif
  
  // Connect the tasks for modified variables, updating the comp
  // map with each modify and doing this in task order.
  // Also do a type check
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    addDependencyEdges(task, task->getRequires(), comps, false);
    addDependencyEdges(task, task->getModifies(), comps, true);
    // Used here just to warn if a modifies comes before its computes
    // in the order that tasks were added to the graph.
    task->visited = true;
  }

  // Initialize variables on the tasks
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    task->visited=false;
    task->sorted=false;
  }
} // end setupTaskConnections()

void TaskGraph::addDependencyEdges(Task* task, Task::Dependency* req,
				  CompMap& comps, bool modifies)
{
  for(; req != 0; req=req->next){
    if(req->dw != Task::OldDW){
      // If DW is finalized, we assume that we already have it,
      // or that we will get it sent to us.  Otherwise, we set
      // up an edge to connect this req to a comp
      pair<CompMap::iterator,CompMap::iterator> iters 
	= comps.equal_range(req->var);
      int count=0;
      for(CompMap::iterator compiter = iters.first;
	  compiter != iters.second; ++compiter){
	if(req->var->typeDescription() != compiter->first->typeDescription())
	  throw TypeMismatchException("Type mismatch for variable: "+req->var->getName());
	// Make sure that we get the comp from the reduction task
	bool add=false;
	if(task->isReductionTask()){
	  // Only those that overlap
	  if(overlaps(compiter->second, req))
	    add=true;
	} else if(req->var->typeDescription()->isReductionVariable()){
	  if(compiter->second->task->isReductionTask())
	    add=true;
	} else {
	  if(overlaps(compiter->second, req))
	    add=true;
	}
	if(add){
	  Task::Dependency* comp = compiter->second;
	  
	  if (modifies) {
	    // not just requires, but modifies, so the comps map must be
	    // updated so future modifies or requires will link to this one.
	    compiter->second = req;

	    // Add dependency edges to each task that requires the data
	    // before it is modified.
	    for (Task::Edge* otherEdge = comp->req_head; otherEdge != 0;
		 otherEdge = otherEdge->reqNext) {
	      Task::Dependency* priorReq =
		const_cast<Task::Dependency*>(otherEdge->req);
	      Task::Edge* edge = scinew Task::Edge(priorReq, req);
	      edges.push_back(edge);
	      req->addComp(edge);
	      priorReq->addReq(edge);
	      if(dbg.active()){
		dbg << "Creating edge from task: " << *priorReq->task << " to task: " << *req->task << '\n';
		dbg << "Prior Req=" << *priorReq << '\n';
		dbg << "Modify=" << *req << '\n';
	      }
      
	    }
	  }
	  
	  Task::Edge* edge = scinew Task::Edge(comp, req);
	  edges.push_back(edge);
	  req->addComp(edge);
	  comp->addReq(edge);
	  
	  if (!edge->comp->task->visited &&
	      !edge->comp->task->isReductionTask()) {
	    cerr << "\nWARNING: A task, '" << task->getName() << "', that ";
	    if (modifies)
	      cerr << "modifies '";
	    else
	      cerr << "requires '";
	    cerr << req->var->getName() << "' was added before computing task";
	    cerr << ", '" << edge->comp->task->getName() << "'"
		 << endl << endl;
	  }
	  count++;
	  if(dbg.active()){
	    dbg << "Creating edge from task: " << *comp->task << " to task: " << *task << '\n';
	    dbg << "Req=" << *req << '\n';
	    dbg << "Comp=" << *comp << '\n';
	  }
	}
      }
      if(count == 0 && (!req->matls || req->matls->size() > 0) 
	 && (!req->patches || req->patches->size() > 0)){
	if(req->patches){
	  cerr << req->patches->size() << "Patches: ";
	  for(int i=0;i<req->patches->size();i++)
	    cerr << req->patches->get(i)->getID() << " ";
	  cerr << '\n';
	} else {
	  cerr << "Patches from task: ";
	  const PatchSet* patches = task->getPatchSet();
	  for(int i=0;i<patches->size();i++){
	    const PatchSubset* pat=patches->getSubset(i);
	    for(int i=0;i<pat->size();i++)
	      cerr << pat->get(i)->getID() << " ";
	    cerr << " ";
	  }
	  cerr << '\n';
	}
	if(req->matls){
	  cerr << req->matls->size() << "Matls: ";
	  for(int i=0;i<req->matls->size();i++)
	    cerr << req->matls->get(i) << " ";
	  cerr << '\n';
	} else {
	  cerr << "Matls from task: ";
	  const MaterialSet* matls = task->getMaterialSet();
	  for(int i=0;i<matls->size();i++){
	    const MaterialSubset* mat = matls->getSubset(i);
	    for(int i=0;i<mat->size();i++)
	      cerr << mat->get(i) << " ";
	    cerr << " ";
	  }
	  cerr << '\n';
	}
	throw InternalError("Scheduler could not find specific production for variable: "+req->var->getName()+", required for task: "+task->getName());
      }
    }
  }
}

void
TaskGraph::processTask(Task* task, vector<Task*>& sortedTasks) const
{
  if(dbg.active())
    dbg << "Looking at task: " << task->getName() << '\n';

  if(task->visited){
    ostringstream error;
    error << "Cycle detected in task graph: already did\n\t"
	  << task->getName();
    error << "\n";
    throw InternalError(error.str());
  }

  task->visited=true;
   
  processDependencies(task, task->getRequires(), sortedTasks);
  processDependencies(task, task->getModifies(), sortedTasks);

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  task->sorted=true;
  if(dbg.active())
    dbg << "Added task: " << task->getName() << '\n';
} // end processTask()


void TaskGraph::processDependencies(Task* task, Task::Dependency* req,
				    vector<Task*>& sortedTasks) const

{
  for(; req != 0; req=req->next){
    if(!req->dw == Task::OldDW){
      Task::Edge* edge = req->comp_head;
#if 0
      if(edge && edge->compNext){
	// more than one compute for a require
	if (!task->isReductionTask())
	  // Only let reduction tasks require a multiple computed
	  // variable. We may wish to change this in the future but
	  // it isn't supported now.
	  throw InternalError(string("Only reduction tasks may require a variable that has multiple computes.\n'") + task->getName() + "' is therefore invalid.\n");
      }
#endif
      
      for (;edge != 0; edge = edge->compNext){
	Task* vtask = edge->comp->task;
	if(!vtask->sorted){
	  if(vtask->visited){
	    ostringstream error;
	    error << "Cycle detected in task graph: trying to do\n\t"
		  << task->getName();
	    error << "\nbut already did:\n\t"
		  << vtask->getName();
	    error << ",\nwhile looking for variable: \n\t" 
		  << req->var->getName();
	    error << "\n";
	    throw InternalError(error.str());
	  }
	  processTask(vtask, sortedTasks);
	}
      }
    }
  }
}

void
TaskGraph::nullSort( vector<Task*>& tasks )
{
  // setupTaskConnections also creates the reduction tasks...
  setupTaskConnections();

  vector<Task*>::iterator iter;

  // No longer going to sort them... let the MixedScheduler take care
  // of calling the tasks when all dependencies are satisfied.
  // Sorting the tasks causes problem because now tasks (actually task
  // groups) run in different orders on different MPI processes.

  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    tasks.push_back( *iter );
  }
}

void
TaskGraph::topologicalSort(vector<Task*>& sortedTasks)
{
  setupTaskConnections();

  vector<Task*>::iterator iter;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if(!task->sorted){
      processTask(task, sortedTasks);
    }
  }
}

void
TaskGraph::addTask(Task* task, const PatchSet* patchset,
		   const MaterialSet* matlset)
{
  task->setSets(patchset, matlset);
  if((patchset && patchset->totalsize() == 0)
     || (matlset && matlset->totalsize() == 0)){
    delete task;
    if(dbg.active())
      dbg << "Killing empty task: " << *task << "\n";
  } else {
    d_tasks.push_back(task);
    if(dbg.active())
      dbg << "Adding task: " << *task << "\n";
  }

  //  maintain d_initRequires and d_initRequiredVars
  for(Task::Dependency* req = task->getRequires(); req != 0; req=req->next){
    if(req->dw == Task::OldDW) {
      d_initRequires.push_back(req);
      d_initRequiredVars.insert(req->var);
    }
  }
}

void
TaskGraph::createDetailedTask(DetailedTasks* tasks, Task* task,
			      const PatchSubset* patches,
			      const MaterialSubset* matls)
{
  DetailedTask* dt = scinew DetailedTask(task, patches, matls, tasks);
  tasks->add(dt);
}

DetailedTasks*
TaskGraph::createDetailedTasks( const ProcessorGroup* pg,
				LoadBalancer* lb,
			        bool useInternalDeps )
{
  vector<Task*> sorted_tasks;
  topologicalSort(sorted_tasks);

  // WARNING - this just grabs ONE level.  For AMR, we may need to be
  // more careful.
  const Level* level = 0;
  for(int i=0;!level && i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->patch_set;
    if(ps && task->matl_set){
      for(int p=0;!level && p<ps->size();p++){
	const PatchSubset* pss = ps->getSubset(p);
	for(int s=0;!level && s<pss->size();s++){
	  const Patch* patch = pss->get(s);
	  level = patch->getLevel();
	}
      }
    }
  }
  ASSERT(level != 0);
  lb->createNeighborhood(level, pg);

  DetailedTasks* dt = scinew DetailedTasks( pg, this, useInternalDeps );
  for(int i=0;i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->patch_set;
    const MaterialSet* ms = task->matl_set;
    if(ps && ms){
      for(int p=0;p<ps->size();p++){
	const PatchSubset* pss = ps->getSubset(p);
	for(int s=0;s<pss->size();s++)
	  ASSERT(pss->get(s)->getLevel() == level);
	for(int m=0;m<ms->size();m++){
	  const MaterialSubset* mss = ms->getSubset(m);
	  if(lb->inNeighborhood(pss, mss))
	    createDetailedTask(dt, task, pss, mss);
	}
      }
    } else if(!ps && !ms){
      createDetailedTask(dt, task, 0, 0);
    } else if(!ps){
      throw InternalError("Task has MaterialSet, but no PatchSet");
    } else {
      throw InternalError("Task has PatchSet, but no MaterialSet");
    }
  }
  return dt;
}

#ifdef __GNUG__
namespace std {
  template<class T> class hash {
  public:
    hash() {}
    unsigned int operator()(const T& str){
      unsigned int h=0;
      for(typename T::const_iterator iter = str.begin(); iter != str.end(); iter++){
	h=(h>>1)^(h<<1);
	h+=*iter;
      }
      return h;
    }
  };
}
#endif

namespace Uintah {
  
class CompTable {
  struct Data {
    Data* next;
    DetailedTask* task;
    Task::Dependency* comp;
    const Patch* patch;
    int matl;
    unsigned int hash;
    
    Data(DetailedTask* task, Task::Dependency* comp,
	 const Patch* patch, int matl)
      : task(task), comp(comp), patch(patch), matl(matl)
    {
      if(patch){
	std::hash<string> h;
	hash=(unsigned int)(((unsigned int)comp->dw<<3)
			    ^(h(comp->var->getName()))
			    ^(patch->getID()<<4)
			    ^matl);
      } else {
	std::hash<string> h;
	hash=(unsigned int)(((unsigned int)comp->dw<<3)
			    ^(h(comp->var->getName()))
			    ^matl);
      }
    }
    ~Data()
    {
    }
    bool operator==(const Data& c) {
      return matl == c.matl && patch == c.patch &&
	comp->dw == c.comp->dw && comp->var->equals(c.comp->var);
    }
  };
  FastHashTable<Data> data;
 public:
  CompTable();
  ~CompTable();
  void remembercomp(DetailedTask* task, Task::Dependency* comp,
		    const PatchSubset* patches, const MaterialSubset* matls);
  bool findcomp(Task::Dependency* req, const Patch* patch, int matlIndex,
		DetailedTask*& dt, Task::Dependency*& comp);
  bool findReplaceComp(Task::Dependency* req, const Patch* patch,
		       int matlIndex,
		       DetailedTask* newTask, Task::Dependency* newComp,
		       DetailedTask*& foundTask, Task::Dependency*& foundComp);
  
};

}

CompTable::CompTable()
{
}

CompTable::~CompTable()
{
}

void CompTable::remembercomp(DetailedTask* task, Task::Dependency* comp,
			     const PatchSubset* patches, const MaterialSubset* matls)
{
  if(patches && matls){
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m=0;m<matls->size();m++){
	int matl = matls->get(m);
	data.insert(new Data(task, comp, patch, matl));
      }
    }
  } 
  else if (matls) {
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      data.insert(new Data(task, comp, 0, matl));
    }
  }
  else if (patches) {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      data.insert(new Data(task, comp, patch, 0));
    }
  }
  else {
    data.insert(new Data(task, comp, 0, 0));
  }
}

bool CompTable::findcomp(Task::Dependency* req, const Patch* patch,
			 int matlIndex, DetailedTask*& dt,
			 Task::Dependency*& comp)
{
  Data key(0, req, patch, matlIndex);
  Data* result;
  if(data.lookup(&key, result)){
    dt=result->task;
    comp=result->comp;
    return true;
  } else {
    return false;
  }
}

bool
CompTable::findReplaceComp(Task::Dependency* req, const Patch* patch,
			   int matlIndex, DetailedTask* newTask,
			   Task::Dependency* newComp, DetailedTask*& foundTask,
			   Task::Dependency*& foundComp)
{
  Data key(0, req, patch, matlIndex);
  Data* result;
  if(data.lookup(&key, result)){
    foundTask=result->task;
    foundComp=result->comp;
    data.remove(&key);
    data.insert(new Data(newTask, newComp, patch, matlIndex));
    return true;
  } else {
    return false;
  }
}


// Will need to do something about const-ness here.
template<class T>
const ComputeSubset<T>* intersection(const ComputeSubset<T>* s1,
				     const ComputeSubset<T>* s2)
{
  if(!s1)
    return s2;
  if(!s2)
    return s1;

  ComputeSubset<T>* result = scinew ComputeSubset<T>;

  if(s1->size() == 0 || s2->size() == 0)
    return result;
  T el1 = s1->get(0);
  for(int i=1;i<s1->size();i++){
    T el = s1->get(i);
    if(el <= el1){
      cerr << "Set not sorted: " << el1 << ", " << el << '\n';
    }
    el1=el;
  }
  T el2 = s2->get(0);
  for(int i=1;i<s2->size();i++){
    T el = s2->get(i);
    if(el <= el2){
      cerr << "Set not sorted: " << el2 << ", " << el << '\n';
    }
    el2=el;
  }
  int i1=0;
  int i2=0;
  for(;;){
    if(s1->get(i1) == s2->get(i2)){
      result->add(s1->get(i1));
      i1++; i2++;
    } else if(s1->get(i1) < s2->get(i2)){
      i1++;
    } else {
      i2++;
    }
    if(i1 == s1->size() || i2 == s2->size())
      break;
  }
  return result;
}

void
TaskGraph::createDetailedDependencies(DetailedTasks* dt, LoadBalancer* lb,
				      const ProcessorGroup* pg)
{
  // We could fuse both of these into one loop, since we know that the
  // taskgraph is already sorted.  It may make it slightly faster

  // Collect all of the comps
  CompTable ct;
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);
    for(Task::Dependency* comp = task->task->getComputes();
	comp != 0; comp = comp->next){
      const PatchSubset* patches;
      switch(comp->patches_dom){
      case Task::NormalDomain:
	patches = intersection(comp->patches, task->patches);
        break;
      case Task::OutOfDomain:
	patches = comp->patches;
	break;
      default:
	throw InternalError("Unknown patches_dom type");
      }
      const MaterialSubset* matls;
      switch(comp->matls_dom){
      case Task::NormalDomain:
	matls = intersection(comp->matls, task->matls);
	break;
      case Task::OutOfDomain:
	matls = comp->matls;
	break;
      default:
	throw InternalError("Unknown matls_dom type");
      }
      if(!patches) {
	// Reduction task
	if (matls && !matls->empty()) {
	  ct.remembercomp(task, comp, 0, matls);
	} else { 
	  ct.remembercomp(task, comp, 0, 0);
	}
      }
      else if(!patches->empty() && !matls->empty())
	ct.remembercomp(task, comp, patches, matls);
      
      if(patches && patches->getReferenceCount() == 0)
	delete patches;
      if(matls && matls->getReferenceCount() == 0)
	delete matls;
    }
  }

  // Put internal links between the reduction tasks so a mixed thread/mpi
  // scheduler won't have out of order reduction problems.
  DetailedTask* lastReductionTask = 0;
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);
    if (task->task->getType() == Task::Reduction) {
      if (lastReductionTask != 0)
	task->addInternalDependency(lastReductionTask);
      lastReductionTask = task;
    }
  }

  int me = pg->myrank();
  // Go through the modifies and find and replace the matching comp
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);
    if(task->task->getType() == Task::Reduction) {
      // only internal dependencies need to be generated for reductions
#if 0
      for (Task::Dependency* req = task->task->getRequires();
	   req != 0; req = req->next) {
	DetailedTask* creator;
	Task::Dependency* comp;
	const PatchSubset* patches = req->patches;
	const MaterialSubset* matls = req->matls;
	if(patches && !patches->empty() && matls && !matls->empty()){
	  for(int i=0;i<patches->size();i++){
	    for(int m=0;m<matls->size();m++){
	      const Patch* patch = patches->get(i);
	      int matl = matls->get(m);
	      bool didFind = ct.findcomp(req, patch, matl, creator, comp);
	      if(!didFind) {
		cerr << "Failure finding " << *req << " for " 
		     << task->getTask()->getName() << endl; 
		throw InternalError("Failed to find comp for dep!");
	      }
	      if(task->getAssignedResourceIndex() == 
		 creator->getAssignedResourceIndex()) {
		task->addInternalDependency(creator);
	      }
	    }
	  }
	}
	else
	  throw InternalError("TaskGraph::createDetailedDependencies, reduction task dependency not supported without patches and materials");
      }
#endif
      continue;
    }

    if(dbg.active() && (task->task->getRequires() != 0))
      dbg << "Looking at requires of detailed task: " << *task << '\n';

    createDetailedDependencies(dt, lb, pg, task, task->task->getRequires(),
			       ct, false);

    if(dbg.active() && (task->task->getModifies() != 0))
      dbg << "Looking at modifies of detailed task: " << *task << '\n';

    createDetailedDependencies(dt, lb, pg, task, task->task->getModifies(),
			       ct, true);
  }
    
  if(dbg.active())
    dbg << "Done creating detailed tasks\n";

}


void
TaskGraph::createDetailedDependencies(DetailedTasks* dt, LoadBalancer* lb,
				      const ProcessorGroup* pg,
				      DetailedTask* task,
				      Task::Dependency* req, CompTable& ct,
				      bool modifies)

{
  for( ; req != 0; req = req->next){
    if(dbg.active())
      dbg << "req: " << *req << '\n';
    
    const PatchSubset* patches;
    switch(req->patches_dom){
    case Task::NormalDomain:
      patches = intersection(req->patches, task->patches);
      break;
    case Task::OutOfDomain:
      patches = req->patches;
      break;
    default:
      throw InternalError("Unknown patches_dom type");
    }
    const MaterialSubset* matls;
    switch(req->matls_dom){
    case Task::NormalDomain:
      matls = intersection(req->matls, task->matls);
      break;
    case Task::OutOfDomain:
      matls = req->matls;
      break;
    default:
      throw InternalError("Unknown matls_dom type");
    }
    if(patches)
      patches->addReference();
    if(matls)
      matls->addReference();
    if(patches && !patches->empty() && matls && !matls->empty()){
      for(int i=0;i<patches->size();i++){
	const Patch* patch = patches->get(i);
	Level::selectType neighbors;
	IntVector low, high;
	patch->computeVariableExtents(req->var->typeDescription()->getType(),
				      req->gtype, req->numGhostCells,
				      neighbors, low, high);
	if(dbg.active()){
	  dbg << "Creating dependency on " << neighbors.size() << " neighbors\n";
	  dbg << "Low=" << low << ", high=" << high << ", var=" << req->var->getName() << '\n';
	}
	for(int i=0;i<neighbors.size();i++){
	  const Patch* neighbor=neighbors[i];
	  if(!lb->inNeighborhood(neighbor))
	    continue;
	  for(int m=0;m<matls->size();m++){
	    int matl = matls->get(m);
	    DetailedTask* creator;
	    Task::Dependency* comp = 0;
	    if(req->dw == Task::OldDW){
	      int proc = findVariableLocation(lb, pg, req, neighbor, matl);
	      creator = dt->getOldDWSendTask(proc);
	      comp=0;
	    } else {
	      bool didFind;
	      if (modifies)
		didFind = ct.findReplaceComp(req, neighbor, matl, task, req,
					     creator, comp);
	      else
		didFind = ct.findcomp(req, neighbor, matl, creator, comp);
	      if(!didFind) {
		cerr << "Failure finding " << *req << " for " << *task<< endl;
		cerr << "creator=" << *creator << '\n';
		cerr << "neighbor=" << *neighbor << '\n';
		cerr << "me=" << pg->myrank() << '\n';
		//throw InternalError("Failed to find comp for dep!");
		continue;
	      }
	    }
	    IntVector l = Max(neighbor->getNodeLowIndex(), low);
	    IntVector h = Min(neighbor->getNodeHighIndex(), high);
	    dt->possiblyCreateDependency(creator, comp, neighbor,
					 task, req, patch,
					 matl, l, h);
	    if (modifies) {
	      // Add links to the modifying task from anything requiring
	      // the variable before it's modified so that it never gets
	      // modified before it gets used.
	      for (DependencyBatch* batch = creator->getComputes(); batch != 0;
		   batch = batch->comp_next) {
		for (DetailedDep* dep = batch->head; dep != 0; 
		     dep = dep->next) {
		  if (dep->req->var == req->var) {
		    list<DetailedTask*>::iterator toTaskIter;
		    for (toTaskIter = dep->toTasks.begin(); 
			 toTaskIter != dep->toTasks.end(); toTaskIter++) {
		      DetailedTask* toTask = *toTaskIter;
		      // dep requires what is to be modified before it is to be
		      // modified so create a dependency between them so the
		      // modifying won't conflist with the previous require.
		      if (dbg.active()) {
			dbg << "Requires to modifies dependency from "
			    << toTask->getTask()->getName()
			    << " to " << task->getTask()->getName() << endl;
		      }
		      dt->possiblyCreateDependency(toTask, 0, 0, task, req, 0,
						   matl, l, h);
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    else if (!patches && matls && !matls->empty()) {
      // requiring reduction variables
      for (int m=0;m<matls->size();m++){
	int matl = matls->get(m);
	DetailedTask* creator;
	Task::Dependency* comp = 0;
	bool didFind = ct.findcomp(req, 0, matl, creator, comp);
	if(!didFind) {
	  cerr << "Failure finding " << *req << " for " 
	       << task->getTask()->getName() << endl; 
	  throw InternalError("Failed to find comp for dep!");
	}
	if(task->getAssignedResourceIndex() == 
	   creator->getAssignedResourceIndex()) {
	  task->addInternalDependency(creator);
	}
      }
    }
    else
      throw InternalError("TaskGraph::createDetailedDependencies, task dependency not supported without patches and materials");
    if(patches && patches->removeReference())
      delete patches;
    if(matls && matls->removeReference())
      delete matls;
  }
}

int TaskGraph::findVariableLocation(LoadBalancer* lb,
				    const ProcessorGroup* pg,
				    Task::Dependency*/* req*/,
				    const Patch* patch, int /*matl*/)
{
  // This needs to be improved, especially for re-distribution on
  // restart from checkpoint.
  int proc = lb->getPatchwiseProcessorAssignment(patch, pg);
  return proc;
}

int TaskGraph::getNumTasks() const
{
   return (int)d_tasks.size();
}

Task* TaskGraph::getTask(int idx)
{
   return d_tasks[idx];
}

TaskGraph::VarLabelMaterialMap* TaskGraph::makeVarLabelMaterialMap()
{
   VarLabelMaterialMap* result = scinew VarLabelMaterialMap;

   for(int i=0;i<(int)d_tasks.size();i++){
     Task* task = d_tasks[i];
     for(Task::Dependency* comp = task->getComputes();
	 comp != 0; comp=comp->next){
       // assume all patches will compute the same labels on the same
       // materials
       const VarLabel* label = comp->var;
       list<int>& matls = (*result)[label->getName()];
       const MaterialSubset* msubset = comp->matls;
       if(msubset){
	 for(int mm=0;mm<msubset->size();mm++){
	   matls.push_back(msubset->get(mm));
	 }
       } else if(label->typeDescription()->getType() == TypeDescription::ReductionVariable) {
	 // Default to material -1 (global)
	 matls.push_back(-1);
       } else {
	 const MaterialSet* ms = task->getMaterialSet();
	 for(int m=0;m<ms->size();m++){
	   const MaterialSubset* msubset = ms->getSubset(m);
	   for(int mm=0;mm<msubset->size();mm++){
	     matls.push_back(msubset->get(mm));
	   }
	 }
       }
     }
   }
   return result;
}

