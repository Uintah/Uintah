
/* REFERENCED */
static char *cvid="$Id$";

/*
 *  ConditionVariable: Condition variable primitive
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Thread.h>

namespace SCICore {
    namespace Thread {
	struct ConditionVariable_private {
	    int num_waiters;
	    Mutex mutex;
	    Semaphore semaphore;
	    ConditionVariable_private();
	    ~ConditionVariable_private();
	};
    }
}

SCICore::Thread::ConditionVariable_private::ConditionVariable_private()
    : num_waiters(0), mutex("Condition variable lock"),
      semaphore("Condition variable semaphore", 0)
{
}

SCICore::Thread::ConditionVariable_private::~ConditionVariable_private()
{
}

SCICore::Thread::ConditionVariable::ConditionVariable(const char* name)
    : d_name(name)
{
    d_priv=new ConditionVariable_private();
}

SCICore::Thread::ConditionVariable::~ConditionVariable()
{
    delete d_priv;
}

void
SCICore::Thread::ConditionVariable::wait(Mutex& m)
{
    d_priv->mutex.lock();
    d_priv->num_waiters++;
    d_priv->mutex.unlock();
    m.unlock();
    // Block until woken up by signal or broadcast
    int s=Thread::couldBlock(d_name);
    d_priv->semaphore.down();
    Thread::couldBlockDone(s);
    m.lock();
}

void
SCICore::Thread::ConditionVariable::conditionSignal()
{
    d_priv->mutex.lock();
    if(d_priv->num_waiters > 0){
        d_priv->num_waiters--;
        d_priv->semaphore.up();
    }
    d_priv->mutex.unlock();
}

void
SCICore::Thread::ConditionVariable::conditionBroadcast()
{
    d_priv->mutex.lock();
    while(d_priv->num_waiters > 0){
        d_priv->num_waiters--;
        d_priv->semaphore.up();
    }
    d_priv->mutex.unlock();
}

//
// $Log$
// Revision 1.2  1999/08/25 22:36:01  sparker
// More thread library updates - now compiles
//
// Revision 1.1  1999/08/25 19:00:47  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:55  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
