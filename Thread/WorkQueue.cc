
#include "WorkQueue.h"
#include "Thread.h"
#include <iostream.h>
extern "C" {
#include <sys/pmo.h>
#include <fetchop.h>
}
#include <stdio.h>

/*
 * Doles out work assignment to various worker threads.  Simple
 * attempts are made at evenly distributing the workload.
 * Initially, assignments are relatively large, and will get smaller
 * towards the end in an effort to equalize the total effort.
 */
static fetchop_reservoir_t reservoir;

struct WorkQueue_private {
    WorkQueue_private();
    fetchop_var_t* pvar;
};

WorkQueue_private::WorkQueue_private()
{
    if(!reservoir){
	reservoir=fetchop_init(USE_DEFAULT_PM, 10);
	if(!reservoir){
	    perror("fetchop_init");
	    Thread::niceAbort();
	}
    }
    pvar=fetchop_alloc(reservoir);
    if(!pvar){
	perror("fetchop_alloc");
	Thread::niceAbort();
    }
    storeop_store(pvar, 0);
}

void WorkQueue::init() {
    storeop_store(priv->pvar, 0);
    if(totalAssignments > nallocated){
	if(assignments)
	    delete[] assignments;
	nallocated=totalAssignments;
	assignments=new int[totalAssignments+1];
    }
    int current_assignment=0;
    int current_assignmentsize=(2*totalAssignments)/(nthreads*(granularity+1));
    int decrement=current_assignmentsize/granularity;
    if(current_assignmentsize==0)
	current_assignmentsize=1;
    if(decrement==0)
	decrement=1;
    int idx=0;
    for(int i=0;i<granularity;i++){
	for(int j=0;j<nthreads;j++){
	    assignments[idx++]=current_assignment;
	    current_assignment+=current_assignmentsize;
	    if(current_assignment >= totalAssignments){
		break;
	    }
	}
	current_assignmentsize-=decrement;
	if(current_assignmentsize<1)
	    current_assignmentsize=1;
    }
    while(current_assignment < totalAssignments){
	assignments[idx++]=current_assignment;
	current_assignment+=current_assignmentsize;
    }
    nassignments=idx;
    assignments[nassignments]=totalAssignments;
    nwaiting=0;
    done=false;
}

WorkQueue::WorkQueue(const char* name, int totalAssignments, int nthreads,
	      bool dynamic, int granularity)
     : name(name), nthreads(nthreads),
       dynamic(dynamic),
       totalAssignments(totalAssignments), granularity(granularity),
       nallocated(0), assignments(0)
{
    priv=new WorkQueue_private();
    //init();
}

WorkQueue::WorkQueue(const WorkQueue& copy)
    : name(copy.name),
      nthreads(copy.nthreads), totalAssignments(copy.totalAssignments),
      dynamic(copy.dynamic), granularity(copy.granularity),
      nallocated(0), assignments(0)
{
    priv=new WorkQueue_private();
    init();
}

WorkQueue::WorkQueue()
    : name(0), nallocated(0), assignments(0)
{
    totalAssignments=0;
    priv=new WorkQueue_private();
}

WorkQueue& WorkQueue::operator=(const WorkQueue& copy)
{
    name=copy.name;
    nthreads=copy.nthreads;
    totalAssignments=copy.totalAssignments;
    dynamic=copy.dynamic;
    granularity=copy.granularity;
    init();
    return *this;
}

WorkQueue::~WorkQueue()
{
    fetchop_free(reservoir, priv->pvar);
    delete priv;
}

bool WorkQueue::nextAssignment(int& start, int& end)
{
    fetchop_var_t i=fetchop_increment(priv->pvar);
    if(i >= nassignments)
	return false;
    start=assignments[i];
    end=assignments[i+1];
    return true;
}

#if 0
void WorkQueue::addWork(int nassignments) {
    if(!dynamic){
        cerr << "ERROR: Work added to a non-dynamic WorkQueue: " << name << '\n';
        return;
    }
    lock.lock();
    int oldn=totalAssignments;
    totalAssignments+=nassignments;
    current_assignmentsize=(current_assignmentsize*totalAssignments)/(oldn+1);
    decrement=(decrement*totalAssignments)/(oldn+1);
    if(nwaiting)
        workdone.cond_broadcast();  // Wake up sleeping workers...
    lock.unlock();
}

void WorkQueue::waitForEmpty() {
    lock.lock();
    while(current_assignment != totalAssignments){
	workdone.wait(lock);
    }
    lock.unlock();
}
#endif
