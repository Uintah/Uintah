
/*
 *  ThreadGroup: A set of threads
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

#include <SCICore/Thread/ThreadGroup.h>
#include <SCICore/Thread/Thread.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define for if(0);else for
#endif

using SCICore::Thread::ThreadGroup;
using SCICore::Thread::Thread;

ThreadGroup* ThreadGroup::s_default_group;
//using std::vector;
using namespace std;

ThreadGroup::ThreadGroup(const char* name, ThreadGroup* parentGroup)
    : d_lock("ThreadGroup lock"), d_name(name), d_parent(parentGroup)
{
    if(parentGroup==0){
        d_parent=s_default_group;
        if(d_parent) // It could still be null if we are making the first one
	    d_parent->addme(this);
    }
}

ThreadGroup::~ThreadGroup()
{
}

int
ThreadGroup::numActive(bool countDaemon)
{
    d_lock.lock();
    int total=0;
    if(countDaemon){
        total=d_threads.size();
    } else {
	for(vector<Thread*>::iterator iter=d_threads.begin();
	    iter != d_threads.end();iter++)
	    if((*iter)->isDaemon())
		total++;
    }
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();
	iter != d_groups.end();iter++)
        total+=(*iter)->numActive(countDaemon);
    d_lock.unlock();
    return total;
}

void
ThreadGroup::stop()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();
	iter != d_groups.end();iter++)
        (*iter)->stop();
    for(vector<Thread*>::iterator iter=d_threads.begin();
	iter != d_threads.end();iter++)
        (*iter)->stop();
    d_lock.unlock();
}

void
ThreadGroup::resume()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();
	iter != d_groups.end();iter++)
        (*iter)->resume();
    for(vector<Thread*>::iterator iter=d_threads.begin();
	iter != d_threads.end();iter++)
        (*iter)->resume();
    d_lock.unlock();
}

void
ThreadGroup::join()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();
	iter != d_groups.end();iter++)
        (*iter)->join();
    for(vector<Thread*>::iterator iter=d_threads.begin();
	iter != d_threads.end();iter++)
        (*iter)->join();
    d_lock.unlock();
}

void
ThreadGroup::detach()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();
	iter != d_groups.end();iter++)
        (*iter)->detach();
    for(vector<Thread*>::iterator iter=d_threads.begin();
	iter != d_threads.end();iter++)
        (*iter)->detach();
    d_lock.unlock();
}

ThreadGroup*
ThreadGroup::parentGroup()
{
    return d_parent;
}

void
ThreadGroup::addme(ThreadGroup* t)
{
    d_lock.lock();
    d_groups.push_back(t);
    d_lock.unlock();
}

void
ThreadGroup::addme(Thread* t)
{
    d_lock.lock();
    d_threads.push_back(t);
    d_lock.unlock();
}

//
// $Log$
// Revision 1.6  1999/10/04 16:49:19  moulding
// changed
// using std::vector;
// to
// using namespace std;
// to help VC compiler
//
// added
// #define for if(0);else for
// for win32
//
// added
// using SCICore::Thread::Thread
// to help VC compiler
//
// Revision 1.5  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.4  1999/08/25 19:00:52  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:38:01  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
