
/*
 *  Barrier: Barrier synchronization primitive (default implementation)
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/RecursiveMutex.h>
#include <Core/Thread/Thread.h>

namespace SCIRun {

struct RecursiveMutex_private {
  Mutex mylock;
  Thread* owner;
  int lock_count;
  RecursiveMutex_private(const char* name);
  ~RecursiveMutex_private();
};

RecursiveMutex_private::RecursiveMutex_private(const char* name)
  : mylock(name)
{
  owner=0;
  lock_count=0;
}

RecursiveMutex_private::~RecursiveMutex_private()
{
}

RecursiveMutex::RecursiveMutex(const char* name)
  : name_(name)
{
  priv_=new RecursiveMutex_private(name);
}

RecursiveMutex::~RecursiveMutex()
{
  delete priv_;
}

void
RecursiveMutex::lock()
{
  int oldstate=Thread::couldBlock(name_);
  Thread* me=Thread::self();
  if(priv_->owner == me){
    priv_->lock_count++;
    return;
  }
  priv_->mylock.lock();
  priv_->owner=me;
  priv_->lock_count=1;
  Thread::couldBlockDone(oldstate);
}

void
RecursiveMutex::unlock()
{
  if(--priv_->lock_count == 0){
    priv_->owner=0;
    priv_->mylock.unlock();
  }
}

} // End namespace SCIRun

