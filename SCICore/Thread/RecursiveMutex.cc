
/* REFERENCED */
static char *id="$Id$";

/*
 *  Barrier.h: Barrier synchronization primitive
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include "RecursiveMutex.h"
#include "Thread.h"

/*
 * Provides a recursive <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
 * <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
 * Nested calls to <b>lock()</b> by the same thread are acceptable,
 * but must be matched with calls to <b>unlock()</b>.  This class
 * may be less efficient that the <b>Mutex</b> class, and should not
 * be used unless the recursive lock feature is really required.
 */

RecursiveMutex::RecursiveMutex(const std::string& name)
    : d_myLock(name)
{
    d_owner=0;
    d_lockCount=0;
}

RecursiveMutex::~RecursiveMutex()
{
}

void RecursiveMutex::lock()
{
    Thread* me=Thread::currentThread();
    if(d_owner == me){
        d_lockCount++;
        return;
    }
    d_myLock.lock();
    d_owner=me;
    d_lockCount=1;
}

void RecursiveMutex::unlock()
{
    if(--d_lockCount == 0){
        d_owner=0;
        d_myLock.unlock();
    }
}

//
// $Log$
// Revision 1.3  1999/08/25 02:37:58  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

