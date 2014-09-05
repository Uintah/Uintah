

#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/DetailedTask.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/NotFinished.h>

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
  d_initreqs.clear();
  edges.clear();
}

template<class T>
bool csoverlaps(const ComputeSubset<T>* s1, const ComputeSubset<T>* s2)
{
  if(s1 == s2)
    return true;
  if(s1->size() == 0 || s2->size() == 0)
    return false;
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
	ReductionTasksMap::iterator it=reductionTasks.find(var);
	if(it == reductionTasks.end()){
	  // No reduction task yet, create one
	  Task* newtask = scinew Task(var->getName()+" reduction",
				      Task::Reduction);
	  newtask->computes(var);
	  reductionTasks[var]=newtask;
	  it = reductionTasks.find(var);
	}
	const PatchSet* ps = task->getPatchSet();
	const MaterialSet* ms = task->getMaterialSet();
	for(int p=0;p<ps->size();p++){
	  const PatchSubset* pss = ps->getSubset(p);
	  for(int m=0;m<ms->size();m++){
	    const MaterialSubset* mss = ms->getSubset(m);
	    it->second->requires(comp->dw, var, pss, mss, Ghost::None);
	  }
	}
      }
    }
  }

  // Add the new reduction tasks to the list of tasks
  for(ReductionTasksMap::iterator it = reductionTasks.begin();
      it != reductionTasks.end(); it++){
    addTask(it->second, 0, 0);
  }

  // Gather the comps for the tasks into a map
  typedef multimap<const VarLabel*, Task::Dependency*, VarLabel::Compare> CompMap;
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

  // Gather a list of requirements
  const Task::reqType& reqs = task->getRequires();
  for(Task::reqType::const_iterator dep = reqs.begin();
      dep != reqs.end(); dep++){
    if(!dep->d_dw->isFinalized()){
      TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
      d_allreqs.insert(artype::value_type(p, dep));
    }
    else
      d_initreqs.push_back(dep);
  }
#else
   NOT_FINISHED("new task stuff (addTask - skip)");
#endif

  // Connect the tasks together using the computes/requires info
  // Also do a type check
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    for(Task::Dependency* req = task->getRequires();
	req != 0; req=req->next){
      if(req->dw != Task::OldDW){
	// If DW is finalized, we assume that we already have it,
	// or that we will get it sent to us.  Otherwise, we set
	// up an edge to connect this req to a comp
	CompMap::iterator compiter = comps.find(req->var);
	if(compiter == comps.end()){
	  cerr << "Vars:\n";
	  for(CompMap::iterator iter=comps.begin();iter != comps.end();iter++)
	    cerr << iter->first->getName() << '\n';
	  throw InternalError("Scheduler could not find production for variable: "+req->var->getName()+", required for task: "+task->getName());
	}
	CompMap::iterator end = comps.upper_bound(req->var);
	int count=0;
	for(;compiter != end; ++compiter){
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
	    Task::Edge* edge = scinew Task::Edge(compiter->second, req);
	    edges.push_back(edge);
	    req->addComp(edge);
	    compiter->second->addReq(edge);
	    count++;
	    dbg << "Creating edge from task: " << *compiter->second->task << " to task: " << *task << '\n';
	    dbg << "Req=" << *req << '\n';
	    dbg << "Comp=" << *compiter->second << '\n';
	  }
	}
	if(count == 0)
	  throw InternalError("Scheduler could not find specific production for variable: "+req->var->getName()+", required for task: "+task->getName());
      }
    }
  }

  // Initialize variables on the tasks
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    task->visited=false;
    task->sorted=false;
  }
} // end setupTaskConnections()

void
TaskGraph::processTask(Task* task, vector<Task*>& sortedTasks) const
{
  dbg << "Looking at task: " << task->getName() << '\n';

  if(task->visited){
    ostringstream error;
    error << "Cycle detected in task graph: already did\n\t"
	  << task->getName();
    error << "\n";
    throw InternalError(error.str());
  }

  task->visited=true;
   
  for(const Task::Dependency* req = task->getRequires();
      req != 0; req=req->next){
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
  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  task->sorted=true;
  dbg << "Added task: " << task->getName() << '\n';
} // end processTask()

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
  d_tasks.push_back(task);

  dbg << "Adding task: " << *task << "\n";
}

void
TaskGraph::createDetailedTask(DetailedTasks* tasks, Task* task,
			      const PatchSubset* patches,
			      const MaterialSubset* matls)
{
  DetailedTask* dt = scinew DetailedTask(task, patches, matls);
  tasks->add(dt);
}

DetailedTasks*
TaskGraph::createDetailedTasks(const ProcessorGroup* pg)
{
  DetailedTasks* dt = scinew DetailedTasks(pg);
  vector<Task*> sorted_tasks;
  topologicalSort(sorted_tasks);

  for(int i=0;i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->patch_set;
    const MaterialSet* ms = task->matl_set;
    if(ps && ms){
      for(int p=0;p<ps->size();p++){
	for(int m=0;m<ms->size();m++){
	  createDetailedTask(dt, task, ps->getSubset(p),
			     ms->getSubset(m));
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

class CompTable {
  struct Data {
    DetailedTask* task;
    Task::Dependency* comp;
    const PatchSubset* patches;
    const MaterialSubset* matls;
    Data(DetailedTask* task, Task::Dependency* comp,
	 const PatchSubset* patches, const MaterialSubset* matls)
      : task(task), comp(comp), patches(patches), matls(matls)
    {
      if(patches)
	patches->addReference();
      if(matls)
	matls->addReference();
    }
    ~Data()
    {
      if(patches && patches->removeReference())
	delete patches;
      if(matls && matls->removeReference())
	delete matls;
    }
  };
  vector<Data*> data;
 public:
  CompTable();
  ~CompTable();
  void remembercomp(DetailedTask* task, Task::Dependency* comp,
		    const PatchSubset* patches, const MaterialSubset* matls);
  bool findcomp(Task::Dependency* req, const Patch* patch, int matlIndex,
		DetailedTask*& dt, Task::Dependency*& comp);
};

CompTable::CompTable()
{
}

CompTable::~CompTable()
{
  for(int i=0;i<(int)data.size();i++)
    delete data[i];
}

void CompTable::remembercomp(DetailedTask* task, Task::Dependency* comp,
			     const PatchSubset* patches, const MaterialSubset* matls)
{
  data.push_back(new Data(task, comp, patches, matls));
}

bool CompTable::findcomp(Task::Dependency* req, const Patch* patch,
			 int matlIndex, DetailedTask*& dt,
			 Task::Dependency*& comp)
{
  for(int i=0;i<(int)data.size();i++){
    Data* d = data[i];
    if(req->var->equals(d->comp->var) && d->patches->contains(patch) &&
       d->matls->contains(matlIndex) && req->dw == d->comp->dw){
      dt=d->task;
      comp=d->comp;
      return true;
    }
  }
  return false;
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
      const PatchSubset* patches = intersection(comp->patches, task->patches);
      const MaterialSubset* matls = intersection(comp->matls, task->matls);
      if(!patches && !matls)
	ct.remembercomp(task, comp, 0, 0); // Reduction task
      else if(!patches->empty() && !matls->empty())
	ct.remembercomp(task, comp, patches, matls);
      else {
	if(patches->getReferenceCount() == 0)
	  delete patches;
	if(matls->getReferenceCount() == 0)
	  delete matls;
      }
    }
  }
  // Go through the reqs and find the matching comp
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);

    // Explicit dependencies are currently not generated for reductions
    if(task->task->getType() == Task::Reduction)
      continue;

    dbg << "Looking at detailed task: " << *task << '\n';
    for(Task::Dependency* req = task->task->getRequires();
	req != 0; req = req->next){
      dbg << "req: " << *req << '\n';

      const PatchSubset* patches = intersection(req->patches, task->patches);
      if(patches)
	patches->addReference();
      const MaterialSubset* matls = intersection(req->matls, task->matls);
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
	  dbg << "Creating dependency on " << neighbors.size() << " neighbors\n";
	  dbg << "Low=" << low << ", high=" << high << ", var=" << req->var->getName() << '\n';
	  for(int i=0;i<neighbors.size();i++){
	    const Patch* neighbor=neighbors[i];
	    for(int m=0;m<matls->size();m++){
	      int matl = matls->get(m);
	      DetailedTask* creator;
	      Task::Dependency* comp;
	      if(req->dw == Task::OldDW){
		int proc = findVariableLocation(lb, pg, req, neighbor, matl);
		creator = dt->getOldDWSendTask(proc);
		comp=0;
	      } else {
		if(!ct.findcomp(req, neighbor, matl, creator, comp))
		  throw InternalError("Failed to find comp for dep!");
	      }
	      IntVector l = Max(neighbor->getNodeLowIndex(), low);
	      IntVector h = Min(neighbor->getNodeHighIndex(), high);
	      dt->possiblyCreateDependency(creator, comp, neighbor,
					   task, req, patch,
					   matl, l, h);
	    }
	  }
	}
      }
      if(patches && patches->removeReference())
	delete patches;
      if(matls && matls->removeReference())
	delete matls;
    }
  }
  dbg << "Done creating detailed tasks\n";
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
	 for(int mm=0;mm<msubset->size();mm++)
	   matls.push_back(msubset->get(mm));
       } else {
	 const MaterialSet* ms = task->getMaterialSet();
	 for(int m=0;m<ms->size();m++){
	   const MaterialSubset* msubset = ms->getSubset(m);
	   for(int mm=0;mm<msubset->size();mm++)
	     matls.push_back(msubset->get(mm));
	 }
       }
     }
   }
   return result;
}
