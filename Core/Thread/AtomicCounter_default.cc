/* REFERENCED */
static char *acid="$Id$";

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

#include <SCICore/Thread/AtomicCounter.h>

SCICore::Thread::AtomicCounter::AtomicCounter(const char* name)
    : d_name(name), d_lock("AtomicCounter lock")
{
}

SCICore::Thread::AtomicCounter::AtomicCounter(const char* name, int value)
    : d_name(name), d_lock("AtomicCounter lock"), d_value(value)
{
}

SCICore::Thread::AtomicCounter::~AtomicCounter()
{
}

SCICore::Thread::AtomicCounter::operator int() const
{
    return d_value;
}

SCICore::Thread::AtomicCounter&
SCICore::Thread::AtomicCounter::operator++()
{
    d_lock.lock();
    ++d_value;
    d_lock.unlock();
    return *this;
}

int
SCICore::Thread::AtomicCounter::operator++(int)
{
    d_lock.lock();
    int ret=d_value++;
    d_lock.unlock();
    return ret;
}

SCICore::Thread::AtomicCounter&
SCICore::Thread::AtomicCounter::operator--()
{
    d_lock.lock();
    --d_value;	
    d_lock.unlock();
    return *this;
}

int
SCICore::Thread::AtomicCounter::operator--(int)
{
    d_lock.lock();
    int ret=d_value--;
    d_lock.unlock();
    return ret;
}

void
SCICore::Thread::AtomicCounter::set(int v)
{
    d_lock.lock();
    d_value=v;
    d_lock.unlock();
}

//
// $Log$
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
