

#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <algorithm>
#include <fstream>
#include <iostream>
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
   vector<Task*>::iterator iter;

   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
      delete *iter;
}

void
TaskGraph::initialize()
{
   vector<Task*>::iterator iter;

   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
      delete *iter;

   d_tasks.clear();
   d_allcomps.clear();
   d_allreqs.clear();
   d_initreqs.clear();
}

map<DependData, int> depToSN;

void
TaskGraph::assignUniqueSerialNumbers()
{
  vector<Task*>::iterator iter;
  int num = 0;

  depToSN.clear();

  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
    Task* task = *iter;

#if DAV_DEBUG
    cerr << "Assigning to task: " << *task << "\n";
#endif
    
    Task::reqType& reqs = task->getRequires();
    for(Task::reqType::iterator dep = reqs.begin();
	dep != reqs.end(); dep++){

      DependData         depData( dep );

      map<DependData, int, DependData>::iterator dToSnIter;
      dToSnIter = depToSN.find( depData );

      if( dToSnIter == depToSN.end() ){
	dep->d_serialNumber = num++;
	depToSN[ depData ] = dep->d_serialNumber;
#if DAV_DEBUG
	cerr << "Dep: " << *dep << "added with serial number:" << dep->d_serialNumber << "\n";
#endif
      } else {
	dep->d_serialNumber = dToSnIter->first.dep->d_serialNumber;
#if 0//DAV_DEBUG
	cerr << "Dep: " << *dep << "already has a serial number, reusing:"
	     << dep->d_serialNumber << "\n";
#endif
      }
    }
  }
} // end assignUniqueSerialNumbers()

void
TaskGraph::assignSerialNumbers()
{
  vector<Task*>::iterator iter;
  int num = 0;

  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
    Task* task = *iter;
    Task::reqType& reqs = task->getRequires();
    for(Task::reqType::iterator dep = reqs.begin();
	dep != reqs.end(); dep++){
      dep->d_serialNumber = num++;
    }
  }
  if(dbg.active()){
    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
      Task* task = *iter;
      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	cerr << *dep << '\n';
      }
    }
  }
  d_maxSerial = num;
} // end assignSerialNumbers()


// setupTaskConnections also adds Reduction Tasks to the graph...
void
TaskGraph::setupTaskConnections()
{
   vector<Task*>::iterator iter;

   // Look for all of the reduction variables - we must treat those
   // special.  Create a fake task that performs the reduction
   // While we are at it, ensure that we aren't producing anything
   // into a frozen data warehouse
   map<const VarLabel*, Task*, VarLabel::Compare> reductionTasks;
   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
      Task* task = *iter;
      if (task->isReductionTask())
	 continue; // already a reduction task so skip it

      const Task::compType& comps = task->getComputes();
      for(Task::compType::const_iterator dep = comps.begin();
	  dep != comps.end(); dep++){
	 if(dep->d_dw->isFinalized()){
	    throw InternalError("Variable produced in old datawarehouse: "+dep->d_var->getName());
	 } else if(dep->d_var->typeDescription()->isReductionVariable()){
	    // Look up this variable in the reductionTasks map
	    const VarLabel* var = dep->d_var;
	    map<const VarLabel*, Task*, VarLabel::Compare>::iterator it=reductionTasks.find(var);
	    if(it == reductionTasks.end()){
	       reductionTasks[var]=scinew Task(var->getName()+" reduction");
	       it = reductionTasks.find(var);
	       it->second->computes(dep->d_dw, var, -1, 0);
	    }
	    if (task->getPatch())
	       it->second->requires(dep->d_dw, var, -1, task->getPatch(),
				    Ghost::None);
	    else
	       it->second->requires(dep->d_dw, var);
	 }
      }
   }

   // Add the new reduction tasks to the list of tasks
   for(map<const VarLabel*, Task*, VarLabel::Compare>::iterator it = reductionTasks.begin();
       it != reductionTasks.end(); it++){
      addTask(it->second);
   }

   // Connect the tasks together using the computes/requires info
   // Also do a type check
   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
      Task* task = *iter;
      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	 if(dep->d_dw->isFinalized()){
#if 0 // Not a valid check for parallel code!

	    if(!dep->d_dw->exists(dep->d_var, dep->d_patch))
	       throw InternalError("Variable required from old datawarehouse, but it does not exist: "+dep->d_var->getFullName(dep->d_matlIndex, dep->d_patch));
#endif
	 } else {
	    TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	    actype::iterator aciter = d_allcomps.find(p);
	    if(aciter == d_allcomps.end())
	       throw InternalError("Scheduler could not find production for variable: "+dep->d_var->getName()+", required for task: "+task->getName());
	    if(dep->d_var->typeDescription() != aciter->first.getLabel()->typeDescription())
	       throw TypeMismatchException("Type mismatch for variable: "+dep->d_var->getName());
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
   dbg << "Looking at task: " << task->getName();
   if(task->getPatch())
      dbg << " on patch " << task->getPatch()->getID();
   dbg << '\n';

   if(task->visited){
      ostringstream error;
      error << "Cycle detected in task graph: already did\n\t"
            << task->getName();
      if(task->getPatch())
	 error << " on patch " << task->getPatch()->getID();
      error << "\n";
      throw InternalError(error.str());
   }

   task->visited=true;
   const Task::reqType& reqs = task->getRequires();
   for(Task::reqType::const_iterator dep = reqs.begin();
       dep != reqs.end(); dep++){
      if(!dep->d_dw->isFinalized()){
	 TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	 pair<actype::const_iterator, actype::const_iterator> range;
	 range = d_allcomps.equal_range(p);
	 actype::const_iterator testiter = range.first;
	 if ((++testiter) != range.second) {
	    // more than one compute for a require
	    if (!task->isReductionTask())
	       // Only let reduction tasks require a multiple computed
	       // variable. We may wish to change this in the future but
	       // it isn't supported now.
	       throw InternalError(string("Only reduction tasks may require a variable that has multiple computes.\n'") + task->getName() + "' is therefore invalid.\n");
	 }
	 
	 for (actype::const_iterator aciter = range.first;
	      aciter != range.second; aciter++) {
	    Task* vtask = aciter->second->d_task;
	    if(!vtask->sorted){
	       if(vtask->visited){
		  ostringstream error;
		  error << "Cycle detected in task graph: trying to do\n\t"
			<< task->getName();
		  if(task->getPatch())
		     error << " on patch " << task->getPatch()->getID();
		  error << "\nbut already did:\n\t"
			<< vtask->getName();
		  if(vtask->getPatch())
		     error << " on patch " << vtask->getPatch()->getID();
		  error << ",\nwhile looking for variable: \n\t" 
			<< dep->d_var->getName() << ", material " 
			<< dep->d_matlIndex;
		  if(dep->d_patch)
		     error << ", patch " << dep->d_patch->getID();
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
   dbg << "Added task: " << task->getName();
   if(task->getPatch())
      dbg << " on patch " << task->getPatch()->getID();
   dbg << '\n';

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
TaskGraph::addTask(Task* task)
{
#if 0// DAV_DEBUG
  cerr << "Adding task: " << *task << "\n";
  cerr << " with: " << task->getRequires().size() << " requires.\n";
  cerr << " with: " << task->getComputes().size() << " computes.\n";
#endif

   d_tasks.push_back(task);
 
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
}

const Task::Dependency* TaskGraph::getComputesForRequires(const Task::Dependency* req)
{
   TaskProduct p(req->d_patch, req->d_matlIndex, req->d_var);
   actype::iterator aciter = d_allcomps.find(p);
   if(aciter == d_allcomps.end())
      throw InternalError("Scheduler could not find production for variable: "+req->d_var->getName());
   return aciter->second;
}

void TaskGraph::getRequiresForComputes(const Task::Dependency* comp,
				       vector<const Task::Dependency*>& reqs)
{
#if 0
   // This REALLY needs to be improved - Steve
   vector<Task*>::iterator iter;
   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
      Task* task = *iter;
      const Task::reqType& deps = task->getRequires();
      Task::reqType::const_iterator dep;
      for (dep = deps.begin(); dep != deps.end(); dep++) {

	 if (!dep->d_dw->isFinalized()) {
	    //extern int myrank;
	    //cerr << myrank << ": " << (dep->d_patch?dep->d_patch->getID():-1) << "/" << (comp->d_patch?comp->d_patch->getID():-1) << ", " << dep->d_matlIndex << "/" << comp->d_matlIndex << ", " << dep->d_var->getName() << "/" << comp->d_var->getName() << ": " << dep->d_task->getName() << '\n';
	    if(dep->d_patch == comp->d_patch 
	       && dep->d_matlIndex == comp->d_matlIndex
	       && (dep->d_var == comp->d_var || dep->d_var->getName() == comp->d_var->getName())){
	       //cerr << myrank << ": " << "MATCH!: " << (dep->d_patch?dep->d_patch->getID():-1) << "/" << (comp->d_patch?comp->d_patch->getID():-1) << ", " << dep->d_matlIndex << "/" << comp->d_matlIndex << ", " << dep->d_var->getName() << "/" << comp->d_var->getName() << ": " << dep->d_task->getName() << '\n';
	       reqs.push_back(dep);
	    }
	 }
      }
   }
   //cerr << "Found " << reqs.size() << " consumers of variable: " << comp->d_var->getName() << " on patch " << comp->d_patch->getID() << " " << comp->d_patch << '\n';
#else
   TaskProduct key(comp->d_patch, comp->d_matlIndex, comp->d_var);
   pair<artype::iterator, artype::iterator> iters;
   iters = d_allreqs.equal_range(key);
   for(artype::iterator iter = iters.first; iter != iters.second; iter++)
      reqs.push_back(iter->second);
#endif
}

bool
TaskGraph::allDependenciesCompleted(Task*) const
{
    //cerr << "TaskGraph::allDependenciesCompleted broken!\n";
    return true;
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
  
   map<TaskProduct, const Task::Dependency*>::iterator it;

   // assume all patches will compute the same labels on the same
   // materials
   for (it = d_allcomps.begin(); it != d_allcomps.end(); it++) {
      const VarLabel* label = (*it).first.getLabel();
      list<int>& matls = (*result)[label->getName()];

      // note that d_allcomps is sorted by materials first, so
      // duplicates in materials will be in order
      int material = (*it).second->d_matlIndex;
      if (matls.size() == 0 || matls.back() != material)
         matls.push_back(material);
   }
  
   return result;
}

bool
DependData::operator==( const DependData & d1 ) const {
  if( ( d1.dep->d_matlIndex      == dep->d_matlIndex ) &&
      ( d1.dep->d_var->getName() == dep->d_var->getName() ) ) {
    if( d1.dep->d_patch && dep->d_patch )
      return ( d1.dep->d_patch->getID() == dep->d_patch->getID() );
    else if( ( d1.dep->d_patch && !dep->d_patch ) ||
	     ( !d1.dep->d_patch && dep->d_patch ) )
      return false;
    else
      return true;
  } else {
    return false;
  }
}

bool
DependData::operator<( const DependData & d ) const {
  if( dep->d_var->getName() > d.dep->d_var->getName() )
    return true;
  else if( dep->d_var->getName() == d.dep->d_var->getName() )
    if( dep->d_matlIndex > d.dep->d_matlIndex )
      return true;
    else if( dep->d_matlIndex == d.dep->d_matlIndex )
      if( dep->d_patch > d.dep->d_patch )
	return true;
      else
	return false;
    else
      return false;
  else
    return false;
}

bool
DependData::operator()( const DependData & d1, const DependData & d2 ) const {

  return d1 < d2;
}

