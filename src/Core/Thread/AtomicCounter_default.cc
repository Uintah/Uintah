
/*
 *  AtomicCounter: Thread-safe integer variable
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/AtomicCounter.h>

namespace SCICore {
    namespace Thread {
	struct AtomicCounter_private {
	    Mutex lock;
	    int value;
	    AtomicCounter_private();
	    ~AtomicCounter_private();
	};
    }
}

using SCICore::Thread::AtomicCounter_private;
using SCICore::Thread::AtomicCounter;

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
}

void
AtomicCounter::set(int v)
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->lock.lock();
    d_priv->value=v;
    d_priv->lock.unlock();
    Thread::couldBlockDone(oldstate);
}

//
// $Log$
// Revision 1.3  1999/08/29 00:47:00  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.2  1999/08/28 03:46:46  sparker
// Final updates before integration with PSE
//
// Revision 1.1  1999/08/25 19:00:46  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:54  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
