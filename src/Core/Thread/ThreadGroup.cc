
#include "ThreadGroup.h"
#include "Thread.h"
#include <string.h>
#include <stdlib.h>

/*
 * A group of threads that are linked together for scheduling
 * and control purposes.  The threads may be stopped, resumed
 * and alerted simultaneously.
 */

void ThreadGroup::addme(ThreadGroup* t) {
    lock.lock();
    ThreadGroup** newgroups=new ThreadGroup*[ngroups+1];
    for(int i=0;i<ngroups;i++)
        newgroups[i]=groups[i];
    newgroups[ngroups++]=t;
    if(groups)
        delete[] groups;
    groups=newgroups;
    lock.unlock();
}

void ThreadGroup::addme(Thread* t) {
    lock.lock();
    Thread** newthreads=new Thread*[nthreads+1];
    for(int i=0;i<nthreads;i++)
        newthreads[i]=threads[i];
    newthreads[nthreads++]=t;
    if(threads)
        delete[] threads;
    threads=newthreads;
    lock.unlock();
}

ThreadGroup* ThreadGroup::default_group;
ThreadGroup::ThreadGroup(char* name, ThreadGroup* parentGroup)
	: lock("ThreadGroup lock"), name(strdup(name)), parent(parentGroup) {
    if(parentGroup==0){
        parent=default_group;
        if(parent) // It could still be null if we are making the first one
    	parent->addme(this);
    }
    ngroups=nthreads=0;
    groups=0;
    threads=0;
}

ThreadGroup::~ThreadGroup() {
    free(name);
}

int ThreadGroup::nactive(bool countDaemon) {
    lock.lock();
    int total;
    if(countDaemon){
        total=nthreads;
    } else {
        for(int i=0;i<nthreads;i++)
    	if(threads[i]->isDaemon())
    	   total++;
    }
    for(int i=0;i<ngroups;i++)
        total+=groups[i]->nactive(countDaemon);
    lock.unlock();
    return total;
}

void ThreadGroup::stop() {
    lock.lock();
    for(int i=0;i<ngroups;i++)
        groups[i]->stop();
    for(int j=0;j<nthreads;j++)
        threads[j]->stop();
    lock.unlock();
}

void ThreadGroup::resume() {
    lock.lock();
    for(int i=0;i<ngroups;i++)
        groups[i]->resume();
    for(int j=0;j<nthreads;j++)
        threads[j]->resume();
    lock.unlock();
}

void ThreadGroup::join() {
    lock.lock();
    for(int i=0;i<ngroups;i++)
        groups[i]->join();
    for(int j=0;j<nthreads;j++)
        threads[j]->join();
    lock.unlock();
}

void ThreadGroup::detach() {
    lock.lock();
    for(int i=0;i<ngroups;i++)
        groups[i]->detach();
    for(int j=0;j<nthreads;j++)
        threads[j]->detach();
    lock.unlock();
}

void ThreadGroup::alert(int code) {
    lock.lock();
    for(int i=0;i<ngroups;i++)
        groups[i]->alert(code);
    for(int j=0;j<nthreads;j++)
        threads[j]->alert(code);
    lock.unlock();
}

ThreadGroup* ThreadGroup::parentGroup() {
    return parent;
}

