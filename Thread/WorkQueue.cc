
#include "WorkQueue.h"
#include <iostream.h>

/*
 * Doles out work assignment to various worker threads.  Simple
 * attempts are made at evenly distributing the workload.
 * Initially, assignments are relatively large, and will get smaller
 * towards the end in an effort to equalize the total effort.
 */

void WorkQueue::init() {
    current_assignment=0;
    current_assignmentsize=(2*totalAssignments)/(nthreads*(granularity+1));
    decrement=current_assignmentsize/granularity;
    if(current_assignmentsize==0)
        current_assignmentsize=1;
    if(decrement==0)
        decrement=1;
    threadcount=0;
    nwaiting=0;
    done=false;
}

WorkQueue::WorkQueue(const char* name, int totalAssignments, int nthreads,
	      bool dynamic, int granularity)
     : name(name), lock("WorkQueue lock"), nthreads(nthreads),
   workdone("WorkQueue done condition"), dynamic(dynamic),
   totalAssignments(totalAssignments), granularity(granularity) {
       init();
}

WorkQueue::WorkQueue(const WorkQueue& copy) : lock("WorkQueue lock"),
  workdone("WorkQueue done condition"), name(copy.name),
  nthreads(copy.nthreads), totalAssignments(copy.totalAssignments),
  dynamic(copy.dynamic), granularity(copy.granularity)
{
    init();
}

WorkQueue::WorkQueue() : name(0), lock("WorkQueue lock"),
  workdone("WorkQueue done condition") {
    current_assignment=totalAssignments=0;
}

WorkQueue& WorkQueue::operator=(const WorkQueue& copy) {
    name=copy.name;
    nthreads=copy.nthreads;
    totalAssignments=copy.totalAssignments;
    dynamic=copy.dynamic;
    granularity=copy.granularity;
    init();
    return *this;
}

WorkQueue::~WorkQueue() {}

bool WorkQueue::nextAssignment(int& start, int& end) {
    lock.lock();
    if(current_assignment == totalAssignments){
        if(!dynamic || done){
    	lock.unlock();
    	return false;
        }
        nwaiting++;
        if(nwaiting == nthreads){
    	done=true;
    	workdone.cond_broadcast();
        } else {
    	workdone.wait(lock);
    	nwaiting--;
        }
        if(done){
    	lock.unlock();
    	return false;
        }
        // Do another assignment...
    }
    start=current_assignment;
    end=current_assignment+current_assignmentsize;
    if(end > totalAssignments)
        end=totalAssignments;
    current_assignment=end;
    if(++threadcount == nthreads){
        threadcount=0;
        current_assignmentsize-=decrement;
        if(current_assignmentsize<1)
    	current_assignmentsize=1;
    }
    lock.unlock();
    return true;
}

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

