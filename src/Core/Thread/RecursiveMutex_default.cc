
/*
 *  Barrier: Barrier synchronization primitive (default implementation)
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

#include <SCICore/Thread/RecursiveMutex.h>
#include <SCICore/Thread/Thread.h>

namespace SCICore {
    namespace Thread {
	struct RecursiveMutex_private {
	    Mutex mylock;
	    Thread* owner;
	    int lock_count;
	    RecursiveMutex_private(const char* name);
	    ~RecursiveMutex_private();
	};
    }
}

using SCICore::Thread::RecursiveMutex_private;
using SCICore::Thread::RecursiveMutex;

SCICore::Thread::RecursiveMutex_private::RecursiveMutex_private(const char* name)
    : mylock(name)
{
    owner=0;
    lock_count=0;
}

SCICore::Thread::RecursiveMutex_private::~RecursiveMutex_private()
{
}

SCICore::Thread::RecursiveMutex::RecursiveMutex(const char* name)
    : d_name(name)
{
    d_priv=new RecursiveMutex_private(name);
}

SCICore::Thread::RecursiveMutex::~RecursiveMutex()
{
    delete d_priv;
}

void
SCICore::Thread::RecursiveMutex::lock()
{
    int oldstate=Thread::couldBlock(d_name);
    Thread* me=Thread::self();
    if(d_priv->owner == me){
        d_priv->lock_count++;
        return;
    }
    d_priv->mylock.lock();
    d_priv->owner=me;
    d_priv->lock_count=1;
    Thread::couldBlockDone(oldstate);
}

void
SCICore::Thread::RecursiveMutex::unlock()
{
    if(--d_priv->lock_count == 0){
        d_priv->owner=0;
        d_priv->mylock.unlock();
    }
}

//
// $Log$
// Revision 1.2  1999/08/28 03:46:49  sparker
// Final updates before integration with PSE
//
// Revision 1.1  1999/08/25 19:00:50  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:58  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
