/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/BrainDamagedScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/ThreadPool.h>
#include <SCICore/Malloc/Allocator.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace Uintah;

using SCICore::Exceptions::InternalError;
using SCICore::Thread::SimpleReducer;
using SCICore::Thread::Thread;
using SCICore::Thread::Time;
using SCICore::Thread::ThreadPool;
using namespace std;

BrainDamagedScheduler::BrainDamagedScheduler( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
    d_numThreads=0;
    d_reducer = 
        scinew SimpleReducer("BrainDamagedScheduler only barrier/reducer");
    d_pool = scinew ThreadPool("BrainDamagedScheduler worker threads");
}

BrainDamagedScheduler::~BrainDamagedScheduler()
{
    vector<TaskRecord*>::iterator iter;

    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
	delete *iter;

    delete d_reducer;
    delete d_pool;
}

void
BrainDamagedScheduler::setNumThreads(int nt)
{
    d_numThreads = nt;
}

void
BrainDamagedScheduler::initialize()
{
    vector<TaskRecord*>::iterator iter;

    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
	delete *iter;

    d_tasks.clear();
    d_targets.clear();
    d_allcomps.clear();
}

void
BrainDamagedScheduler::setupTaskConnections()
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
	    it->second->requires(dep->d_dw, var, 0, dep->d_patch, Ghost::None);
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
	       throw InternalError("Scheduler could not find production for variable: "+dep->d_var->getName());
	    if(dep->d_var->typeDescription() != aciter->first.getLabel()->typeDescription())
	       throw TypeMismatchException("Type mismatch for variable: "+dep->d_var->getName());
	 }
      }
   }
}

void
BrainDamagedScheduler::performTask(TaskRecord* task,
				   const ProcessorContext * pc) const
{
   //   cerr << "Looking at task: " << task->task->getName() << " on patch " << (void*)task->task->getPatch()->getID() << '\n';
   if(task->visited)
      throw InternalError("Cycle detected in task graph");
   task->visited=true;
   const vector<Task::Dependency*>& reqs = task->task->getRequires();
   for(vector<Task::Dependency*>::const_iterator iter = reqs.begin();
       iter != reqs.end(); iter++){
      Task::Dependency* dep = *iter;
      OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw.get_rep());
      if(!dw->isFinalized()){
	 TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	 map<TaskProduct, TaskRecord*>::const_iterator aciter = d_allcomps.find(p);
	 if(!aciter->second->task->isCompleted())
	    performTask(aciter->second, pc);
      }
   }

#if 0
   double start = Time::currentSeconds();
#endif
   task->task->doit(pc);
#if 0
   double dt = Time::currentSeconds()-start;
   cout << "Completed task: " << task->task->getName() << " on patch: " << task->task->getPatch()->getID() << " (" << dt << " seconds)\n";
#endif
}

void
BrainDamagedScheduler::execute(const ProcessorContext * pc,
			             DataWarehouseP   & dwp )
{
    if(d_tasks.size() == 0){
	cerr << "WARNING: Scheduler executed, but no tasks\n";
	return;
    }
    setupTaskConnections();

    dumpDependencies();

    vector<TaskRecord*>::iterator iter;
    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
       TaskRecord* task = *iter;
       if(!task->task->isCompleted()){

	 const Patch * patch = task->task->getPatch();

	 // Figure out which MPI node should be doing this task.
	 // need to actually figure it out... THIS IS A HACK
	 int taskLocation;
	 if( !patch || patch->getID() >= 0 && patch->getID() <= 2 )
	   taskLocation = 0;
	 else if( patch->getID() >= 3 && patch->getID() <= 11 )
	   taskLocation = 1;
	 else
	    throw InternalError("BrainDamagedScheduler: Unknown patch");		                              
	 // If it is this node, then kick off the task.
	 // Either way, record which node is doing the task so
	 // so that if one of our tasks needs data from it, we
	 // will know where to go get it from.

	 // Run through all the variables that this task computes and
	 // register them in the DataWarehouse so that the DW will know
	 // where to find them if it needs to at some future point.
	 const vector<Task::Dependency*> & computes = 
		                                  task->task->getComputes();
	 vector<Task::Dependency*>::const_iterator iter = 
		                                  computes.begin();
	 while( iter != computes.end() ) {
	   dwp->registerOwnership( (*iter)->d_var,
				   (*iter)->d_patch,
				   taskLocation );
	   iter++;
	 }

	 if( d_MpiRank == taskLocation ) {  // I am responsible for this task
	   performTask(task, pc);
	 }
       }
    }
	  
#if 0
    int totalcompleted = 0;
    int numThreads = pc->numThreads();
    for(;;){
	int ncompleted=0;
	vector<TaskRecord*>::iterator iter;
	for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
	    
	    TaskRecord* task=*iter;
	    double start = Time::currentSeconds();
	    if(!task->task->isCompleted() && allDependenciesCompleted(task)){

	        // Figure out which MPI node should be doing this task.

	        int taskLocation = d_MpiRank; // need to actually
		                              // figure it out...

	        // If it is this node, then kick off the task.
	        // Either way, record which node is doing the task so
	        // so that if one of our tasks needs data from it, we
	        // will know where to go get it from.

	        if( d_MpiRank == taskLocation ) { 
		  // I am responsible for this task
		  if(task->task->usesThreads()){
		    //cerr << "Performing task with " << numThreads 
		    //     << " threads: " << task->task->getName() << '\n';
                    d_pool->parallel(this,
				     &BrainDamagedScheduler::runThreadedTask,
				     numThreads,
				     task, pc, d_reducer);
		    //task->task->doit(threadpc);
		  } else {
		    //cerr << "Performing task: " << task->task->getName() 
		    //     << '\n';
		    task->task->doit(pc);
		  }
		} else { // I am not responsible for this task, so run
		         // through all the variables that it computes
		         // and register them in the DataWarehouse so
		         // that it will know where to find them if it
		         // needs to at some future point.

		  const vector<Task::Dependency*> & computes = 
		                                     task->task->getComputes();
		  vector<Task::Dependency*>::const_iterator iter = 
		                                              computes.begin();
		  while( iter != computes.end() ) {
		    dwp->registerOwnership( (*iter)->d_var,
					    (*iter)->d_patch,
					    taskLocation );
		    iter++;
		  }
		}
		double dt = Time::currentSeconds()-start;
		cout << "Completed task: " << task->task->getName() << " (" << dt << " seconds)\n";
		ncompleted++;
	    }
	}
	if(ncompleted == 0)
	    throw InternalError("BrainDamagedScheduler stalled");
	totalcompleted += ncompleted;
	if(totalcompleted == d_tasks.size())
	    break;
    }
#endif
    OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dwp.get_rep());;
    dw->finalize();
}

void
BrainDamagedScheduler::addTarget(const VarLabel* target)
{
    d_targets.push_back(target);
}

void
BrainDamagedScheduler::addTask(Task* task)
{
   TaskRecord* tr = scinew TaskRecord(task);
   d_tasks.push_back(tr);
 
   const vector<Task::Dependency*>& comps = task->getComputes();
   for(vector<Task::Dependency*>::const_iterator iter = comps.begin();
       iter != comps.end(); iter++){
      Task::Dependency* dep = *iter;
      if(task->isReductionTask() ||
	  !dep->d_var->typeDescription()->isReductionVariable()){
	 TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	 map<TaskProduct,TaskRecord*>::iterator aciter = d_allcomps.find(p);
	 if(aciter != d_allcomps.end()) {
	    throw InternalError("Two tasks compute the same result: "+dep->d_var->getName());
	 }
	 d_allcomps[p] = tr;
      }
   }
}

bool
BrainDamagedScheduler::allDependenciesCompleted(TaskRecord*) const
{
    //cerr << "BrainDamagedScheduler::allDependenciesCompleted broken!\n";
    return true;
}

DataWarehouseP
BrainDamagedScheduler::createDataWarehouse( int generation )
{
    return scinew OnDemandDataWarehouse( d_MpiRank, d_MpiProcesses, generation );
}

void
BrainDamagedScheduler::runThreadedTask(int threadNumber, TaskRecord* task,
				       const ProcessorContext* pc,
				       SimpleReducer* barrier)
{
    ProcessorContext* subpc = 
               pc->createContext(threadNumber, pc->numThreads(), barrier);
    task->task->doit(subpc);
    delete subpc;
}

void
BrainDamagedScheduler::dumpDependencies()
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
	cerr << "BrainDamagedScheduler::dumpDependencies: unable to open output file!\n";
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

		depfile << task1->getName();
		if(task1->getPatch())
		   depfile << "\\nPatch" << task1->getPatch()->getID();
		depfile << " "  << task2->getName() << "\\nPatch";
		if(task2->getPatch())
		   depfile << task2->getPatch()->getID();
		depfile << endl;
	    }
	}
    }

    depfile.close();
}

BrainDamagedScheduler::
TaskRecord::~TaskRecord()
{
    delete task;
}

BrainDamagedScheduler::
TaskRecord::TaskRecord(Task* t)
    : task(t)
{
   visited=false;
}

//
// $Log$
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
