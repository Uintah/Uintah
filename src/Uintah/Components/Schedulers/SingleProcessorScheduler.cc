/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/DebugStream.h>
#include <SCICore/Thread/Time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <strstream>
#include <unistd.h>

using namespace Uintah;

using SCICore::Exceptions::InternalError;
using SCICore::Util::DebugStream;
using namespace std;
using namespace SCICore::Thread;

static DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
    vector<TaskRecord*>::iterator iter;

    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
	delete *iter;
}

void
SingleProcessorScheduler::initialize()
{
    vector<TaskRecord*>::iterator iter;

    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
	delete *iter;

    d_tasks.clear();
    d_allcomps.clear();
}

void
SingleProcessorScheduler::setupTaskConnections()
{
   // Look for all of the reduction variables - we must treat those
   // special.  Create a fake task that performs the reduction
   // While we are at it, ensure that we aren't producing anything
   // into a frozen data warehouse
   vector<TaskRecord*>::iterator iter;
   map<const VarLabel*, Task*, VarLabel::Compare> reductionTasks;
   for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
      TaskRecord* task = *iter;
      const vector<Task::Dependency*>& comps = task->task->getComputes();
      for(vector<Task::Dependency*>::const_iterator iter = comps.begin();
	  iter != comps.end(); iter++){
	 Task::Dependency* dep = *iter;
	 OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw.get_rep());;
	 if(dw->isFinalized()){
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
	    it->second->requires(dep->d_dw, var, -1, task->task->getPatch(),
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
      TaskRecord* task = *iter;
      const vector<Task::Dependency*>& reqs = task->task->getRequires();
      for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
	  iter != reqs.end(); iter++){
	 Task::Dependency* dep = *iter;
	 OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw.get_rep());;
	 if(dw->isFinalized()){
	    if(!dw->exists(dep->d_var, dep->d_patch))
	       throw InternalError("Variable required from old datawarehouse, but it does not exist: "+dep->d_var->getName());
	 } else {
	    TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	    map<TaskProduct, TaskRecord*>::iterator aciter = d_allcomps.find(p);
	    if(aciter == d_allcomps.end())
	       throw InternalError("Scheduler could not find production for variable: "+dep->d_var->getName()+", required for task: "+task->task->getName());
	    if(dep->d_var->typeDescription() != aciter->first.getLabel()->typeDescription())
	       throw TypeMismatchException("Type mismatch for variable: "+dep->d_var->getName());
	 }
      }
   }
}

void
SingleProcessorScheduler::performTask(TaskRecord* task,
				   const ProcessorContext * pc) const
{
   dbg << "Looking at task: " << task->task->getName();
   if(task->task->getPatch())
      dbg << " on patch " << task->task->getPatch()->getID();
   dbg << '\n';
   if(task->visited){
      ostrstream error;
      error << "Cycle detected in task graph: already did\n\t"
            << task->task->getName() << " on patch "
            << task->task->getPatch()->getID() << "\n";
      throw InternalError(error.str());
   }

   task->visited=true;
   const vector<Task::Dependency*>& reqs = task->task->getRequires();
   for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
       iter != reqs.end(); iter++){
      Task::Dependency* dep = *iter;
      OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw.get_rep());
      if(!dw->isFinalized()){
	 TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	 map<TaskProduct, TaskRecord*>::const_iterator aciter = d_allcomps.find(p);
	 if(!aciter->second->task->isCompleted()){
	   if(aciter->second->visited){
	     ostrstream error;
	     error << "Cycle detected in task graph: trying to do\n\t"
		   << task->task->getName() << " on patch "
		   << task->task->getPatch()->getID()
		   << "\nbut already did:\n\t"
		   << aciter->second->task->getName() << " on patch "
		   << aciter->second->task->getPatch()->getID()
		   << ",\nwhile looking for variable: \n\t" 
		   << dep->d_var->getName() << ", material " 
		   << dep->d_matlIndex << ", patch " << dep->d_patch->getID()
		   << "\n";
	     throw InternalError(error.str());
	   }
	   performTask(aciter->second, pc);
	 }
      }
   }

   double start = Time::currentSeconds();
   task->task->doit(pc);
   double dt = Time::currentSeconds()-start;
   dbg << "Completed task: " << task->task->getName();
   if(task->task->getPatch())
      dbg << " on patch " << task->task->getPatch()->getID();
   dbg << " (" << dt << " seconds)\n";
}

void
SingleProcessorScheduler::execute(const ProcessorContext * pc,
			             DataWarehouseP   & dwp )
{
    if(d_tasks.size() == 0){
	cerr << "WARNING: Scheduler executed, but no tasks\n";
	return;
    }
    dbg << "Executing " << d_tasks.size() << " tasks\n";
    setupTaskConnections();
    dbg << "After setup, there are " << d_tasks.size() << " tasks\n";

    dumpDependencies();

    vector<TaskRecord*>::iterator iter;
    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
       TaskRecord* task = *iter;
       if(!task->task->isCompleted()){
	 performTask(task, pc);
       }
    }
    OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dwp.get_rep());;
    dw->finalize();
}

void
SingleProcessorScheduler::addTask(Task* task)
{
   TaskRecord* tr = scinew TaskRecord(task);
   d_tasks.push_back(tr);
 
   const vector<Task::Dependency*>& comps = task->getComputes();
   for(vector<Task::Dependency*>::const_iterator iter = comps.begin();
       iter != comps.end(); iter++){
      Task::Dependency* dep = *iter;
      TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
      map<TaskProduct,TaskRecord*>::iterator aciter = d_allcomps.find(p);
      if(aciter != d_allcomps.end()) 
	 throw InternalError("Two tasks compute the same result: "+dep->d_var->getName()+" (tasks: "+task->getName()+" and "+aciter->second->task->getName()+")");
      d_allcomps[p] = tr;
   }
}

bool
SingleProcessorScheduler::allDependenciesCompleted(TaskRecord*) const
{
    //cerr << "SingleProcessorScheduler::allDependenciesCompleted broken!\n";
    return true;
}

DataWarehouseP
SingleProcessorScheduler::createDataWarehouse( int generation )
{
    return scinew OnDemandDataWarehouse( d_MpiRank, d_MpiProcesses, generation );
}

void
SingleProcessorScheduler::dumpDependencies()
{
    static int call_nr = 0;
    
    // the first call is just some initialization tasks. All subsequent calls
    // will have the same dependency graph, modulo an output task. The first
    // non-initialization call will have the output task, so we'll just output
    // that one.
    if (call_nr++ != 1)
    	return;
	
    ofstream depfile("dependencies");
    if (!depfile) {
	cerr << "SingleProcessorScheduler::dumpDependencies: unable to open output file!\n";
	return;	// dependency dump failure shouldn't be fatal to anything else
    }

    vector<TaskRecord*>::const_iterator iter;
    for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    	const TaskRecord* taskrec = *iter;

	const vector<Task::Dependency*>& deps = taskrec->task->getRequires();
	vector<Task::Dependency*>::const_iterator dep_iter;
	for (dep_iter = deps.begin(); dep_iter != deps.end(); dep_iter++) {
	    const Task::Dependency* dep = *dep_iter;

	    OnDemandDataWarehouse* dw =
	    	dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw.get_rep());
	    if (!dw->isFinalized()) {

		TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
		map<TaskProduct, TaskRecord*>::const_iterator deptask =
	    	    d_allcomps.find(p);

		const Task* task1 = taskrec->task;
		const Task* task2 = deptask->second->task;

		depfile << "\"" << task1->getName();
		if(task1->getPatch())
		   depfile << "\\nPatch" << task1->getPatch()->getID();
		depfile << "\" \""  << task2->getName();
		if(task2->getPatch())
		   depfile << "\\nPatch" << task2->getPatch()->getID();
		depfile << "\"" << endl;
	    }
	}
    }

    depfile.close();
}

SingleProcessorScheduler::
TaskRecord::~TaskRecord()
{
    delete task;
}

SingleProcessorScheduler::
TaskRecord::TaskRecord(Task* t)
    : task(t)
{
   visited=false;
}

//
// $Log$
// Revision 1.2  2000/06/16 22:59:39  guilkey
// Expanded "cycle detected" print statement
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.20  2000/06/15 21:57:11  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.19  2000/06/14 23:43:25  jehall
// - Made "cycle detected" exception more informative
//
// Revision 1.18  2000/06/08 17:11:39  jehall
// - Added quotes around task names so names with spaces are parsable
//
// Revision 1.17  2000/06/03 05:27:23  sparker
// Fixed dependency analysis for reduction variables
// Removed warnings
// Now allow for task patch to be null
// Changed DataWarehouse emit code
//
// Revision 1.16  2000/05/30 20:19:22  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.15  2000/05/30 17:09:37  dav
// MPI stuff
//
// Revision 1.14  2000/05/21 20:10:48  sparker
// Fixed memory leak
// Added scinew to help trace down memory leak
// Commented out ghost cell logic to speed up code until the gc stuff
//    actually works
//
// Revision 1.13  2000/05/19 18:35:09  jehall
// - Added code to dump the task dependencies to a file, which can be made
//   into a pretty dependency graph.
//
// Revision 1.12  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.11  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.10  2000/05/05 06:42:42  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.9  2000/04/28 21:12:04  jas
// Added some includes to get it to compile on linux.
//
// Revision 1.8  2000/04/26 06:48:32  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/20 18:56:25  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/19 21:20:02  dav
// more MPI stuff
//
// Revision 1.5  2000/04/19 05:26:10  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.4  2000/04/11 07:10:40  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 01:03:16  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
