
/*
 *  ThreadPool: A pool of threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Thread/ThreadPool.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <stdio.h>

namespace SCIRun {

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

ThreadPool::ThreadPool(const char* name)
  : name_(name), lock_("ThreadPool lock"), barrier("ThreadPool barrier")
{
  group_ = 0;
}

ThreadPool::~ThreadPool()
{
  // All of the threads will go away with this
  delete group_;
}

void ThreadPool::wait()
{
  barrier.wait(threads_.size()+1);
}

void
ThreadPool::parallel(const ParallelBase& helper, int nthreads)
{
  lock_.lock();
  if(nthreads >= (int)threads_.size()){
    if(!group_)
      group_=new ThreadGroup("Parallel group");
    int oldsize = threads_.size();
    threads_.resize(nthreads);
    for(int i=oldsize;i<nthreads;i++){
      char buf[50];
      sprintf(buf, "Parallel thread %d of %d", i, nthreads);
      threads_[i] = new ThreadPoolHelper(i, this);
      Thread* t = new Thread(threads_[i], buf, group_,
			     Thread::Stopped);
      t->setDaemon(true);
      t->detach();
      t->migrate(i);
      t->resume();
    }
  }
  Thread::self()->migrate(nthreads%Thread::numProcessors());
  for(int i=0;i<nthreads;i++){
    threads_[i]->helper = &helper;
    //	threads_[i]->start_sema.up();
  }
  barrier.wait(nthreads+1);
  barrier.wait(nthreads+1);
  for(int i=0;i<nthreads;i++){
    //threads_[i]->done_sema.down();
    threads_[i]->helper = 0;
  }
  lock_.unlock();
}

} // End namespace SCIRun
