
/* REFERENCED */
static char *bid="$Id$";

/*
 *  Barrier: Barrier synchronization primitive
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/Barrier.h>

struct Barrier_private {
    Mutex mutex;
    ConditionVariable cond0;
    ConditionVariable cond1;
    int cc;
    int nwait;
    Barrier_private();
};

Barrier_private::Barrier_private()
: mutex("Barrier lock"),
  cond0("Barrier condition 0"), cond1("Barrier condition 1"),
  cc(0), nwait(0)
{
}   

Barrier::Barrier(const std::string& name, int numThreads)
 : d_name(name), d_numThreads(numThreads), d_threadGroup(0)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

Barrier::Barrier(const std::string& name, ThreadGroup* threadGroup)
 : d_name(name), d_numThreads(0), d_threadGroup(threadGroup)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

Barrier::~Barrier()
{
    delete d_priv;
}

void
Barrier::wait()
{
    int n=d_threadGroup?d_threadGroup->numActive(true):d_numThreads;
    Thread_private* p=Thread::currentThread()->d_priv;
    int oldstate=push_bstack(p, STATE_BLOCK_SEMAPHORE, d_name.c_str());
    d_priv->mutex.lock();
    ConditionVariable& cond=d_priv->cc?d_priv->cond0:d_priv->cond1;
    /*int me=*/d_priv->nwait++;
    if(d_priv->nwait == n){
	// Wake everybody up...
	d_priv->nwait=0;
	d_priv->cc=1-d_priv->cc;
	cond.conditionBroadcast();
    } else {
	cond.wait(d_priv->mutex);
    }
    d_priv->mutex.unlock();
    pop_bstack(p, oldstate);
}

//
// $Log$
// Revision 1.1  1999/08/25 19:00:47  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
