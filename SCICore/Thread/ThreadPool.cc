
/*
 *  ThreadPool: A pool of threads
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Thread/ThreadPool.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/ThreadGroup.h>
using SCICore::Thread::ParallelBase;
using SCICore::Thread::ThreadPool;
#include <stdio.h>

namespace SCICore {
    namespace Thread {
	class ThreadPoolHelper : public Runnable {
	    const ParallelBase* helper;
	    int proc;
	    friend class ThreadPool;
	    ThreadPool* pool;
	    //Semaphore start_sema;	
	    //Semaphore done_sema;
	public:
	    ThreadPoolHelper(int proc, ThreadPool* pool)
		: helper(0), proc(proc), pool(pool)
		  //start_sema("ThreadPool helper startup semaphore", 0),
		  //done_sema("ThreadPool helper completion semaphore", 0) 
	    {
	    }
	    virtual ~ThreadPoolHelper() {}
	    virtual void run() {
		for(;;){
		    //start_sema.down();
		    pool->wait();
		    ParallelBase* cheat=(ParallelBase*)helper;
		    cheat->run(proc);
		    //done_sema.up();
		    pool->wait();
		}
	    }
	};	
    }
}

ThreadPool::ThreadPool(const char* name)
    : d_name(name), d_lock("ThreadPool lock"), barrier("ThreadPool barrier")
{
    d_group = 0;
}

ThreadPool::~ThreadPool()
{
    // All of the threads will go away with this
    delete d_group;
}

void ThreadPool::wait()
{
    barrier.wait(d_threads.size()+1);
}

void
ThreadPool::parallel(const ParallelBase& helper, int nthreads)
{
    d_lock.lock();
    if(nthreads >= (int)d_threads.size()){
	if(!d_group)
	    d_group=new ThreadGroup("Parallel group");
	int oldsize = d_threads.size();
	d_threads.resize(nthreads);
	for(int i=oldsize;i<nthreads;i++){
	    char buf[50];
	    sprintf(buf, "Parallel thread %d of %d", i, nthreads);
	    d_threads[i] = new ThreadPoolHelper(i, this);
	    Thread* t = new Thread(d_threads[i], buf, d_group,
				   Thread::Stopped);
	    t->setDaemon(true);
	    t->detach();
	    t->migrate(i);
	    t->resume();
	}
    }
    Thread::self()->migrate(nthreads%Thread::numProcessors());
    for(int i=0;i<nthreads;i++){
	d_threads[i]->helper = &helper;
	//	d_threads[i]->start_sema.up();
    }
    barrier.wait(nthreads+1);
    barrier.wait(nthreads+1);
    for(int i=0;i<nthreads;i++){
	//d_threads[i]->done_sema.down();
	d_threads[i]->helper = 0;
    }
    d_lock.unlock();
}
