
/*
 *  WorkQueue: Manage assignments of work
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#include <Core/Thread/WorkQueue.h>
namespace SCIRun {

WorkQueue::WorkQueue(const char* name)
    : name_(name), current_assignment_("WorkQueue counter")
{
}

WorkQueue::~WorkQueue()
{
}

bool
WorkQueue::nextAssignment(int& start, int& end)
{
    int i=current_assignment_++; // Atomic ++
    if(i >= (int)assignments_.size()-1)
	return false;
    start=assignments_[i];
    end=assignments_[i+1];
    return true;
}

void
WorkQueue::refill(int new_ta, int new_nthreads,
				   int new_granularity)
{
    if(new_ta == total_assignments_ && new_nthreads == num_threads_
       && new_granularity == granularity_){
	current_assignment_.set(0);
    } else {
	total_assignments_=new_ta;
	num_threads_=new_nthreads;
	granularity_=new_granularity;
	fill();
    }
}

void
WorkQueue::fill()
{
    current_assignment_.set(0);

    if(total_assignments_==0){
	assignments_.resize(0);
	return;
    }
    assignments_.reserve(total_assignments_+1);
    int current_assignment=0;
    int current_assignmentsize=(2*total_assignments_)/(num_threads_*(granularity_+1));
    int decrement=current_assignmentsize/granularity_;
    if(current_assignmentsize==0)
	current_assignmentsize=1;
    if(decrement==0)
	decrement=1;
    for(int i=0;i<granularity_;i++){
	for(int j=0;j<num_threads_;j++){
	    assignments_.push_back(current_assignment);
	    current_assignment+=current_assignmentsize;
	    if(current_assignment >= total_assignments_){
		break;
	    }
	}
	if(current_assignment >= total_assignments_){
	  break;
	}
	current_assignmentsize-=decrement;
	if(current_assignmentsize<1)
	    current_assignmentsize=1;
	if(current_assignment >= total_assignments_){
	    break;
	}
    }
    while(current_assignment < total_assignments_){
	assignments_.push_back(current_assignment);
	current_assignment+=current_assignmentsize;
    }
    assignments_.push_back(total_assignments_);
    done_=false;
}


} // End namespace SCIRun
