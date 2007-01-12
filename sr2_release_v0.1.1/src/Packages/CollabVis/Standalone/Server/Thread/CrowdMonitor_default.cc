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
 *  CrowdMonitor: Multiple reader/single writer locks
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/CrowdMonitor.h>

namespace SCIRun {

struct CrowdMonitor_private {
  ConditionVariable write_waiters;
  ConditionVariable read_waiters;
  Mutex lock;
  int num_readers_waiting;
  int num_writers_waiting;
  int num_readers;
  int num_writers;
  CrowdMonitor_private();
  ~CrowdMonitor_private();
};

CrowdMonitor_private::CrowdMonitor_private()
  : write_waiters("CrowdMonitor write condition"),
    read_waiters("CrowdMonitor read condition"),
    lock("CrowdMonitor lock")
{
  num_readers_waiting=0;
  num_writers_waiting=0;
  num_readers=0;
  num_writers=0;
}

CrowdMonitor_private::~CrowdMonitor_private()
{
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
  priv_->lock.lock();
  while(priv_->num_writers > 0){
    priv_->num_readers_waiting++;
    int s=Thread::couldBlock(name_);
    priv_->read_waiters.wait(priv_->lock);
    Thread::couldBlockDone(s);
    priv_->num_readers_waiting--;
  }
  priv_->num_readers++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
}

void
CrowdMonitor::readUnlock()
{
  priv_->lock.lock();
  priv_->num_readers--;
  if(priv_->num_readers == 0 && priv_->num_writers_waiting > 0)
    priv_->write_waiters.conditionSignal();
  priv_->lock.unlock();
}

void
CrowdMonitor::writeLock()
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  while(priv_->num_writers || priv_->num_readers){
    // Have to wait...
    priv_->num_writers_waiting++;
    int s=Thread::couldBlock(name_);
    priv_->write_waiters.wait(priv_->lock);
    Thread::couldBlockDone(s);
    priv_->num_writers_waiting--;
  }
  priv_->num_writers++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
} // End namespace SCIRun

void
CrowdMonitor::writeUnlock()
{
  priv_->lock.lock();
  priv_->num_writers--;
  if(priv_->num_writers_waiting)
    priv_->write_waiters.conditionSignal(); // Wake one of them up...
  else if(priv_->num_readers_waiting)
    priv_->read_waiters.conditionBroadcast(); // Wake all of them up...
  priv_->lock.unlock();
}

} // End namespace SCIRun
