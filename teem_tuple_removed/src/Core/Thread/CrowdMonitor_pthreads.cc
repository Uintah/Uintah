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
 *  CrowdMonitor: Multiple reader/single writer locks, implementation
 *   for pthreads simply wrapping pthread_rwlock_t
 *
 *  Written by:
 *   Author:  Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Thread/CrowdMonitor.h>
#include <pthread.h>

/*** WARNING: THIS IMPLEMENTATION HAS NOT BEEN TESTED - 3/6/2002 ***/

namespace SCIRun {

struct CrowdMonitor_private {
  pthread_rwlock_t lock_;
  CrowdMonitor_private();
  ~CrowdMonitor_private();
};

CrowdMonitor_private::CrowdMonitor_private()
{
  pthread_rwlock_init(&lock_, 0);
}

CrowdMonitor_private::~CrowdMonitor_private()
{
  pthread_rwlock_destroy(&lock_);
}

CrowdMonitor::CrowdMonitor(const char* name)
  : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "CrowdMonitor: %s\n", name);
    Thread::initialize();
  }
  priv_=new CrowdMonitor_private;
}

CrowdMonitor::~CrowdMonitor()
{
  delete priv_;
  priv_=0;
}

void
CrowdMonitor::readLock()
{
  int oldstate=Thread::couldBlock(name_);
  pthread_rwlock_rdlock(&priv_->lock_);
  Thread::couldBlockDone(oldstate);
}

void
CrowdMonitor::readUnlock()
{
  pthread_rwlock_unlock(&priv_->lock_);  
}

void
CrowdMonitor::writeLock()
{
  int oldstate=Thread::couldBlock(name_);
  pthread_rwlock_wrlock(&priv_->lock_);
  Thread::couldBlockDone(oldstate);
} // End namespace SCIRun

void
CrowdMonitor::writeUnlock()
{
  pthread_rwlock_unlock(&priv_->lock_);  
}

} // End namespace SCIRun
