
/*
 *  ThreadGroup: A set of threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Thread.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define for if(0);else for
#endif

namespace SCIRun {

ThreadGroup* ThreadGroup::s_default_group;
//using std::vector;
using namespace std;

ThreadGroup::ThreadGroup(const char* name, ThreadGroup* parentGroup)
    : lock_("ThreadGroup lock"), name_(name), parent_(parentGroup)
{
    if(parentGroup==0){
        parent_=s_default_group;
        if(parent_) // It could still be null if we are making the first one
	    parent_->addme(this);
    }
}

ThreadGroup::~ThreadGroup()
{
}

int
ThreadGroup::numActive(bool countDaemon)
{
    lock_.lock();
    int total=0;
    if(countDaemon){
        total=threads_.size();
    } else {
	for(vector<Thread*>::iterator iter=threads_.begin();
	    iter != threads_.end();iter++)
	    if((*iter)->isDaemon())
		total++;
    }
    for(vector<ThreadGroup*>::iterator iter=groups_.begin();
	iter != groups_.end();iter++)
        total+=(*iter)->numActive(countDaemon);
    lock_.unlock();
    return total;
}

void
ThreadGroup::stop()
{
    lock_.lock();
    for(vector<ThreadGroup*>::iterator iter=groups_.begin();
	iter != groups_.end();iter++)
        (*iter)->stop();
    for(vector<Thread*>::iterator iter=threads_.begin();
	iter != threads_.end();iter++)
        (*iter)->stop();
    lock_.unlock();
}

void
ThreadGroup::resume()
{
    lock_.lock();
    for(vector<ThreadGroup*>::iterator iter=groups_.begin();
	iter != groups_.end();iter++)
        (*iter)->resume();
    for(vector<Thread*>::iterator iter=threads_.begin();
	iter != threads_.end();iter++)
        (*iter)->resume();
    lock_.unlock();
}

void
ThreadGroup::join()
{
    lock_.lock();
    for(vector<ThreadGroup*>::iterator iter=groups_.begin();
	iter != groups_.end();iter++)
        (*iter)->join();
    for(vector<Thread*>::iterator iter=threads_.begin();
	iter != threads_.end();iter++)
        (*iter)->join();
    lock_.unlock();
}

void
ThreadGroup::detach()
{
    lock_.lock();
    for(vector<ThreadGroup*>::iterator iter=groups_.begin();
	iter != groups_.end();iter++)
        (*iter)->detach();
    for(vector<Thread*>::iterator iter=threads_.begin();
	iter != threads_.end();iter++)
        (*iter)->detach();
    lock_.unlock();
}

ThreadGroup*
ThreadGroup::parentGroup()
{
    return parent_;
}

void
ThreadGroup::addme(ThreadGroup* t)
{
    lock_.lock();
    groups_.push_back(t);
    lock_.unlock();
}

void
ThreadGroup::addme(Thread* t)
{
    lock_.lock();
    threads_.push_back(t);
    lock_.unlock();
}


} // End namespace SCIRun
