/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/BrainDamagedScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/Task.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/ThreadPool.h>
#include <iostream>
#include <set>

namespace Uintah {
namespace Components {

using SCICore::Exceptions::InternalError;
using SCICore::Thread::SimpleReducer;
using SCICore::Thread::Thread;
using SCICore::Thread::Time;
using SCICore::Thread::ThreadPool;
using std::cerr;
using std::cout;
using std::find;
using std::set;
using std::vector;

BrainDamagedScheduler::BrainDamagedScheduler()
{
    d_numThreads=0;
    d_reducer = 
        new SimpleReducer("BrainDamagedScheduler only barrier/reducer");
    d_pool = new ThreadPool("BrainDamagedScheduler worker threads");
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
}

void
BrainDamagedScheduler::setupTaskConnections()
{
    // Gather all comps...
    vector<Task::Dependency*>      allcomps;
    vector<TaskRecord*>::iterator  iter;

    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
	TaskRecord* task = *iter;
	task->task->addComps(allcomps);
    }

    // Verify that no two comps are the same and verify all comps
    // have zero ghost cells
#if 0
    set<Task::Dependency> unique_comps;
    for(vector<Task::Dependency*>::iterator iter=allcomps.begin();
	iter != allcomps.end(); iter++){
	Task::Dependency* dep = *iter;
	if(dep->numGhostCells != 0)
	    throw SchedulerException("Computed value requires ghost cells");
	if(allcomps_set.find( *dep ) != allcomps_set.end()){
	    throw SchedulerException("Two tasks compute the same result");
	} else {
	    allcomps_set.insert( *dep);
	}
    }

    // For each of the reqs, find the comp(s)
    for( iter=d_tasks.begin(); iter != tasks.end(); iter++){
	TaskRecord* task = *iter;
	vector<Task::Dependency*> reqs;
	task->task->addReqs(reqs);

	vector<Task*> taskDeps;
	for(vector<Task::Dependency*>::iterator iter=reqs.begin();
	    iter != reqs.end(); iter++){
	    Task::Dependency* dep = *iter;
	    vector<Task::Dependency>::iterator search = find(unique_comps.begin(), unique_comps.end(), dep);
	    if(search == unique_comps.end()){
		// Look in the datastore
		if(!dep->dw->exists(dep->varname, dep->region, dep->numGhostCells)){
		    cerr << "not found, task: " << dep->task->getName() << ", dep: " << (*iter)->varname << '\n';
		    throw SchedulerException("Task not found to compute dependency");
		}
	    } else {
		taskDeps.push_back();
	    }
	}
	task->task->setDependencies(taskDeps);
    }

    // Set up reverse dependencies
    for( iter=d_tasks.begin(); iter != tasks.end(); iter++ ){
	TaskRecord* task = *iter;
	task->reverseDeps.clear();
    }
    for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
	TaskRecord* task = *iter;
	for(vector<TaskRecord*>::iterator iter=task->deps.begin();
	    iter != task->deps.end(); iter++){
	    (*iter)->reverseDeps.push_back(task);
	}
    }

    // Perform a type consistency check

    // See if it is a connected set

    // See if there are any circular dependencies
#endif

}

void
BrainDamagedScheduler::execute(const ProcessorContext* pc)
{
    if(d_tasks.size() == 0){
	cerr << "WARNING: Scheduler executed, but no tasks\n";
	return;
    }
    setupTaskConnections();
    int totalcompleted = 0;
    int numThreads = pc->numThreads();
    for(;;){
	int ncompleted=0;
	vector<TaskRecord*>::iterator iter;
	for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ){
	    
	    TaskRecord* task=*iter;
	    double start = Time::currentSeconds();
	    if(!task->task->isCompleted() && allDependenciesCompleted(task)){
		if(task->task->usesThreads()){
		    //cerr << "Performing task with " << numThreads << " threads: " << task->task->getName() << '\n';
                    d_pool->parallel(this, &BrainDamagedScheduler::runThreadedTask,
				   numThreads,
				   task, pc, d_reducer);
//task->task->doit(threadpc);
		} else {
		    //cerr << "Performing task: " << task->task->getName() << '\n';
		    task->task->doit(pc);
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
}

void
BrainDamagedScheduler::addTarget(const VarLabel* target)
{
#if 0
    d_targets.push_back(target);
#else
    cerr << "BrainDamagedScheduler::addTarget not done!\n";
#endif
}

void
BrainDamagedScheduler::addTask(Task* t)
{
    d_tasks.push_back(new TaskRecord(t));
}

bool
BrainDamagedScheduler::allDependenciesCompleted(TaskRecord*) const
{
    //cerr << "BrainDamagedScheduler::allDependenciesCompleted broken!\n";
    return true;
}

DataWarehouseP
BrainDamagedScheduler::createDataWarehouse()
{
    return new OnDemandDataWarehouse();
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

BrainDamagedScheduler::
TaskRecord::~TaskRecord()
{
    delete task;
}

BrainDamagedScheduler::
TaskRecord::TaskRecord(Task* t)
    : task(t)
{
}


} // end namespace Components
} // end namespace Uintah

//
// $Log$
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
