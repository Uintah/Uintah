
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
}

AtomicCounter::operator int() const
{
    return priv_->value;
}

int
AtomicCounter::operator++()
{
    int oldstate=Thread::couldBlock(name_);
    priv_->lock.lock();
    int ret=++priv_->value;
    priv_->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator++(int)
{
    int oldstate=Thread::couldBlock(name_);
    priv_->lock.lock();
    int ret=priv_->value++;
    priv_->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator--()
{
    int oldstate=Thread::couldBlock(name_);
    priv_->lock.lock();
    int ret=--priv_->value;	
    priv_->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator--(int)
{
    int oldstate=Thread::couldBlock(name_);
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
