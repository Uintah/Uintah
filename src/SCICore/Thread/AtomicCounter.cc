/* REFERENCED */
static char *id="$Id$";

/*
 *  AtomicCounter.h: Thread-safe integer variable
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include "AtomicCounter.h"

/*
 * Provides a simple atomic counter.  This will work just like an
 * integer, but guarantees atomicty of the ++ and -- operators.
 * Despite their convenience, you do not want to make a large number
 * of these objects.  See also <b>WorkQueue</b>.
 */

AtomicCounter::AtomicCounter(const std::string& name)
    : d_name(name), d_lock("AtomicCounter lock")
{
}

AtomicCounter::AtomicCounter(const std::string& name, int value)
    : d_name(name), d_lock("AtomicCounter lock"), d_value(value)
{
}

AtomicCounter::~AtomicCounter()
{
}

AtomicCounter::operator int() const
{
    return d_value;
}

AtomicCounter& AtomicCounter::operator++()
{
    d_lock.lock();
    ++d_value;
    d_lock.unlock();
    return *this;
}

int AtomicCounter::operator++(int)
{
    d_lock.lock();
    int ret=d_value++;
    d_lock.unlock();
    return ret;
}

AtomicCounter& AtomicCounter::operator--()
{
    d_lock.lock();
    --d_value;	
    d_lock.unlock();
    return *this;
}

int AtomicCounter::operator--(int)
{
    d_lock.lock();
    int ret=d_value--;
    d_lock.unlock();
    return ret;
}

void AtomicCounter::set(int v)
{
    d_lock.lock();
    d_value=v;
    d_lock.unlock();
}
//
// $Log$
// Revision 1.3  1999/08/25 02:37:54  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

