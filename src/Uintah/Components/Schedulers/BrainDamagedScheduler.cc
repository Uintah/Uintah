
#include <Uintah/Components/Schedulers/BrainDamagedScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Exceptions/SchedulerException.h>
#include <Uintah/Grid/Task.h>
#include <SCICore/Thread/SimpleReducer.h>
using SCICore::Thread::SimpleReducer;
#include <SCICore/Thread/Thread.h>
using SCICore::Thread::Thread;
#include <SCICore/Thread/Time.h>
using SCICore::Thread::Time;
#include <SCICore/Thread/ThreadPool.h>
using SCICore::Thread::ThreadPool;
#include <iostream>
#include <set>
using std::cerr;
using std::cout;
using std::find;
using std::set;
using std::vector;

BrainDamagedScheduler::BrainDamagedScheduler()
{
    numThreads=0;
    reducer = new SimpleReducer("BrainDamagedScheduler only barrier/reducer");
    pool = new ThreadPool("BrainDamagedScheduler worker threads");
}

void BrainDamagedScheduler::setNumThreads(int nt)
{
    numThreads = nt;
}

BrainDamagedScheduler::TaskRecord::~TaskRecord()
{
    delete task;
}

BrainDamagedScheduler::~BrainDamagedScheduler()
{
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++)
	delete *iter;
    delete reducer;
    delete pool;
}

void BrainDamagedScheduler::initialize()
{
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++)
	delete *iter;

    tasks.clear();
    targets.clear();
}

void BrainDamagedScheduler::setupTaskConnections()
{
    // Gather all comps...
    vector<Task::Dependency*> allcomps;
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++){
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
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++){
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
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++){
	TaskRecord* task = *iter;
	task->reverseDeps.clear();
    }
    for(vector<TaskRecord*>::iterator iter=tasks.begin();
	iter != tasks.end(); iter++){
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

void BrainDamagedScheduler::execute(const ProcessorContext* pc)
{
    if(tasks.size() == 0){
	cerr << "WARNING: Scheduler executed, but no tasks\n";
	return;
    }
    setupTaskConnections();
    int totalcompleted = 0;
    int numThreads = pc->numThreads();
    for(;;){
	int ncompleted=0;
	for(vector<TaskRecord*>::iterator iter=tasks.begin();
	    iter != tasks.end(); iter++){
	    
	    TaskRecord* task=*iter;
	    double start = Time::currentSeconds();
	    if(!task->task->isCompleted() && allDependenciesCompleted(task)){
		if(task->task->usesThreads()){
		    //cerr << "Performing task with " << numThreads << " threads: " << task->task->getName() << '\n';
                    pool->parallel(this, &BrainDamagedScheduler::runThreadedTask,
				   numThreads,
				   task, pc, reducer);
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
	    throw SchedulerException("BrainDamagedScheduler stalled");
	totalcompleted += ncompleted;
	if(totalcompleted == tasks.size())
	    break;
    }
}

void BrainDamagedScheduler::addTarget(const std::string& target)
{
    targets.push_back(target);
}

void BrainDamagedScheduler::addTask(Task* t)
{
    tasks.push_back(new TaskRecord(t));
}

bool BrainDamagedScheduler::allDependenciesCompleted(TaskRecord* task) const
{
    //cerr << "BrainDamagedScheduler::allDependenciesCompleted broken!\n";
    return true;
}

BrainDamagedScheduler::TaskRecord::TaskRecord(Task* t)
    : task(t)
{
}

DataWarehouseP BrainDamagedScheduler::createDataWarehouse()
{
    return new OnDemandDataWarehouse();
}

void BrainDamagedScheduler::runThreadedTask(int threadNumber, TaskRecord* task,
					    const ProcessorContext* pc,
					    SimpleReducer* barrier)
{
    ProcessorContext* subpc = pc->createContext(threadNumber, pc->numThreads(), barrier);
    task->task->doit(subpc);
    delete subpc;
}
