
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
}


AtomicCounter_private::AtomicCounter_private()
    : lock("AtomicCounter lock")
{
}

AtomicCounter_private::~AtomicCounter_private()
{
}

AtomicCounter::AtomicCounter(const char* name)
    : d_name(name)
{
    d_priv=new AtomicCounter_private;
}

AtomicCounter::AtomicCounter(const char* name, int value)
    : d_name(name)
{
    d_priv=new AtomicCounter_private;
    d_priv->value=value;
}

AtomicCounter::~AtomicCounter()
{
    delete d_priv;
}

AtomicCounter::operator int() const
{
    return d_priv->value;
}

int
AtomicCounter::operator++()
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    int ret=++d_priv->value;
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator++(int)
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    int ret=d_priv->value++;
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator--()
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    int ret=--d_priv->value;	
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
}

int
AtomicCounter::operator--(int)
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    int ret=d_priv->value--;
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);
    return ret;
} // End namespace SCIRun

void
AtomicCounter::set(int v)
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    d_priv->value=v;
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);

