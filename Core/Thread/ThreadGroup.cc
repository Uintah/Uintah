
#include "ThreadGroup.h"
#include "Thread.h"
#include <string.h>
#include <stdlib.h>

/*
 * A group of threads that are linked together for scheduling
 * and control purposes.  The threads may be stopped, resumed
 * and alerted simultaneously.
 */

void ThreadGroup::addme(ThreadGroup* t)
{
    d_lock.lock();
    d_groups.push_back(t);
    d_lock.unlock();
}

void ThreadGroup::addme(Thread* t)
{
    d_lock.lock();
    d_threads.push_back(t);
    d_lock.unlock();
}

ThreadGroup* ThreadGroup::s_defaultGroup;

ThreadGroup::ThreadGroup(const std::string& name, ThreadGroup* parentGroup)
    : d_lock("ThreadGroup lock"), d_name(name), d_parent(parentGroup)
{
    if(parentGroup==0){
        d_parent=s_defaultGroup;
        if(d_parent) // It could still be null if we are making the first one
	    d_parent->addme(this);
    }
}

ThreadGroup::~ThreadGroup()
{
}

int ThreadGroup::numActive(bool countDaemon)
{
    d_lock.lock();
    int total=0;
    if(countDaemon){
        total=d_threads.size();
    } else {
	for(vector<Thread*>::iterator iter=d_threads.begin();iter != d_threads.end();iter++)
	    if((*iter)->isDaemon())
		total++;
    }
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();iter != d_groups.end();iter++)
        total+=(*iter)->numActive(countDaemon);
    d_lock.unlock();
    return total;
}

void ThreadGroup::stop() {
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();iter != d_groups.end();iter++)
        (*iter)->stop();
    for(vector<Thread*>::iterator iter=d_threads.begin();iter != d_threads.end();iter++)
        (*iter)->stop();
    d_lock.unlock();
}

void ThreadGroup::resume()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();iter != d_groups.end();iter++)
        (*iter)->resume();
    for(vector<Thread*>::iterator iter=d_threads.begin();iter != d_threads.end();iter++)
        (*iter)->resume();
    d_lock.unlock();
}

void ThreadGroup::join()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();iter != d_groups.end();iter++)
        (*iter)->join();
    for(vector<Thread*>::iterator iter=d_threads.begin();iter != d_threads.end();iter++)
        (*iter)->join();
    d_lock.unlock();
}

void ThreadGroup::detach()
{
    d_lock.lock();
    for(vector<ThreadGroup*>::iterator iter=d_groups.begin();iter != d_groups.end();iter++)
        (*iter)->detach();
    for(vector<Thread*>::iterator iter=d_threads.begin();iter != d_threads.end();iter++)
        (*iter)->detach();
    d_lock.unlock();
}

ThreadGroup* ThreadGroup::parentGroup()
{
    return d_parent;
}

