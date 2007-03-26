/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "RecursiveMutex: %s\n", name);
    Thread::initialize();
  }
  priv_=new RecursiveMutex_private(name);
}

RecursiveMutex::~RecursiveMutex()
{
  delete priv_;
  priv_=0;
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

