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
 *  AtomicCounter: Thread-safe integer variable
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {
struct AtomicCounter_private {
  Mutex lock;
  int value;
  AtomicCounter_private();
  ~AtomicCounter_private();
};
}

using SCIRun::AtomicCounter_private;
using SCIRun::AtomicCounter;

AtomicCounter_private::AtomicCounter_private()
    : lock("AtomicCounter lock")
{
}

AtomicCounter_private::~AtomicCounter_private()
{
}

AtomicCounter::AtomicCounter(const char* name)
    : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "AtomicCounter: %s\n", name);
    Thread::initialize();
  }
  priv_=new AtomicCounter_private;
}

AtomicCounter::AtomicCounter(const char* name, int value)
    : name_(name)
{
  priv_=new AtomicCounter_private;
  priv_->value=value;
}

AtomicCounter::~AtomicCounter()
{
  delete priv_;
  priv_=0;
}

AtomicCounter::operator int() const
{
    return priv_->value;
}

int
AtomicCounter::operator++()
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=++priv_->value;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator++(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=priv_->value++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator--()
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=--priv_->value;	
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator--(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=priv_->value--;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
} 

void
AtomicCounter::set(int v)
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  priv_->value=v;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
}
