
/*
 *  WorkQueue: Manage assignments of work
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#include <SCICore/Thread/WorkQueue.h>

using SCICore::Thread::WorkQueue;

WorkQueue::WorkQueue(const char* name)
    : d_name(name), d_current_assignment("WorkQueue counter")
{
}

WorkQueue::~WorkQueue()
{
}

bool
WorkQueue::nextAssignment(int& start, int& end)
{
    int i=d_current_assignment++; // Atomic ++
    if(i >= (int)d_assignments.size()-1)
	return false;
    start=d_assignments[i];
    end=d_assignments[i+1];
    return true;
}

void
WorkQueue::refill(int new_ta, int new_nthreads,
				   int new_granularity)
{
    if(new_ta == d_total_assignments && new_nthreads == d_num_threads
       && new_granularity == d_granularity){
	d_current_assignment.set(0);
    } else {
	d_total_assignments=new_ta;
	d_num_threads=new_nthreads;
	d_granularity=new_granularity;
	fill();
    }
}

void
WorkQueue::fill()
{
    d_current_assignment.set(0);

    if(d_total_assignments==0){
	d_assignments.resize(0);
	return;
    }
    d_assignments.reserve(d_total_assignments+1);
    int current_assignment=0;
    int current_assignmentsize=(2*d_total_assignments)/(d_num_threads*(d_granularity+1));
    int decrement=current_assignmentsize/d_granularity;
    if(current_assignmentsize==0)
	current_assignmentsize=1;
    if(decrement==0)
	decrement=1;
    for(int i=0;i<d_granularity;i++){
	for(int j=0;j<d_num_threads;j++){
	    d_assignments.push_back(current_assignment);
	    current_assignment+=current_assignmentsize;
	    if(current_assignment >= d_total_assignments){
		break;
	    }
	}
	if(current_assignment >= d_total_assignments){
	  break;
	}
	current_assignmentsize-=decrement;
	if(current_assignmentsize<1)
	    current_assignmentsize=1;
	if(current_assignment >= d_total_assignments){
	    break;
	}
    }
    while(current_assignment < d_total_assignments){
	d_assignments.push_back(current_assignment);
	current_assignment+=current_assignmentsize;
    }
    d_assignments.push_back(d_total_assignments);
    d_done=false;
}

//
// $Log$
// Revision 1.5  1999/08/28 03:46:53  sparker
// Final updates before integration with PSE
//
// Revision 1.4  1999/08/25 19:00:53  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
