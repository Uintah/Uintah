
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
  : d_name(name)
{
  d_priv=new RecursiveMutex_private(name);
}

RecursiveMutex::~RecursiveMutex()
{
  delete d_priv;
}

void
RecursiveMutex::lock()
{
  int oldstate=Thread::couldBlock(d_name);
  Thread* me=Thread::self();
  if(d_priv->owner == me){
    d_priv->lock_count++;
    return;
  }
  d_priv->mylock.lock();
  d_priv->owner=me;
  d_priv->lock_count=1;
  Thread::couldBlockDone(oldstate);
}

void
RecursiveMutex::unlock()
{
  if(--d_priv->lock_count == 0){
    d_priv->owner=0;
    d_priv->mylock.unlock();
  }
}

} // End namespace SCIRun

