
// $Id$

#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/TypeDescription.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/DebugStream.h>
#include <PSECore/XMLUtil/XMLUtil.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

using namespace Uintah;

using SCICore::Exceptions::InternalError;
using namespace PSECore::XMLUtil;
using namespace std;
static SCICore::Util::DebugStream dbg("TaskGraph", false);

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
}

void
TaskGraph::assignSerialNumbers()
{
   vector<Task*>::iterator iter;
   int num = 0;

   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
      Task* task = *iter;
      const vector<Task::Dependency*>& reqs = task->getRequires();
      for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
	  iter != reqs.end(); iter++){
	 Task::Dependency* dep = *iter;
	 dep->d_serialNumber = num++;
      }
   }
}

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
      const vector<Task::Dependency*>& comps = task->getComputes();
      for(vector<Task::Dependency*>::const_iterator iter = comps.begin();
	  iter != comps.end(); iter++){
	 Task::Dependency* dep = *iter;
	 if(dep->d_dw->isFinalized()){
	    throw InternalError("Variable produced in old datawarehouse: "+dep->d_var->getName());
	 } else if(dep->d_var->typeDescription()->isReductionVariable()){
	    // Look up this variable in the reductionTasks map
	    const VarLabel* var = dep->d_var;
	    map<const VarLabel*, Task*, VarLabel::Compare>::iterator it=reductionTasks.find(var);
	    if(it == reductionTasks.end()){
	       reductionTasks[var]=new Task(var->getName()+" reduction");
	       it = reductionTasks.find(var);
	       it->second->computes(dep->d_dw, var, -1, 0);
	    }
	    it->second->requires(dep->d_dw, var, -1, task->getPatch(),
				 Ghost::None);
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
      const vector<Task::Dependency*>& reqs = task->getRequires();
      for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
	  iter != reqs.end(); iter++){
	 Task::Dependency* dep = *iter;
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
}

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
   const vector<Task::Dependency*>& reqs = task->getRequires();
   for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
       iter != reqs.end(); iter++){
      Task::Dependency* dep = *iter;
      if(!dep->d_dw->isFinalized()){
	 TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	 actype::const_iterator aciter = d_allcomps.find(p);
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
   // All prerequisites are done - add this task to the list
   sortedTasks.push_back(task);
   task->sorted=true;
   dbg << "Added task: " << task->getName();
   if(task->getPatch())
      dbg << " on patch " << task->getPatch()->getID();
   dbg << '\n';
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
   d_tasks.push_back(task);
 
   const vector<Task::Dependency*>& comps = task->getComputes();
   for(vector<Task::Dependency*>::const_iterator iter = comps.begin();
       iter != comps.end(); iter++){
      Task::Dependency* dep = *iter;
      TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
      actype::iterator aciter = d_allcomps.find(p);
      if(aciter != d_allcomps.end()) 
	 throw InternalError("Two tasks compute the same result: "+dep->d_var->getName()+" (tasks: "+task->getName()+" and "+aciter->second->d_task->getName()+")");
      d_allcomps[p] = dep;
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
   // This REALLY needs to be improved - Steve
   vector<Task*>::iterator iter;
   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
      Task* task = *iter;
      const vector<Task::Dependency*>& deps = task->getRequires();
      vector<Task::Dependency*>::const_iterator dep_iter;
      for (dep_iter = deps.begin(); dep_iter != deps.end(); dep_iter++) {
	 const Task::Dependency* dep = *dep_iter;

	 if (!dep->d_dw->isFinalized()) {
	    extern int myrank;
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

//
// $Log$
// Revision 1.5  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.4  2000/07/25 20:59:28  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.3  2000/07/19 21:47:59  jehall
// - Changed task graph output to XML format for future extensibility
// - Added statistical information about tasks to task graph output
//
// Revision 1.2  2000/06/27 14:59:53  jehall
// - Removed extra call to dumpDependencies()
//
// Revision 1.1  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//
