
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

#include <Core/Thread/Barrier.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {
	struct Barrier_private {
	    Mutex mutex;
	    ConditionVariable cond0;
	    ConditionVariable cond1;
	    int cc;
	    int nwait;
	    Barrier_private();
	    ~Barrier_private();
	};
    }
}


Barrier_private::Barrier_private()
    : mutex("Barrier lock"),
      cond0("Barrier condition 0"), cond1("Barrier condition 1"),
      cc(0), nwait(0)
{
}

Barrier_private::~Barrier_private()
{
}

Barrier::Barrier(const char* name)
    : d_name(name)
{
    d_priv=new Barrier_private;
}

Barrier::~Barrier()
{
    delete d_priv;
}

void
Barrier::wait(int n)
{
    int oldstate=Thread::couldBlock(d_name);
    d_priv->mutex.lock();
    ConditionVariable& cond=d_priv->cc?d_priv->cond0:d_priv->cond1;
    d_priv->nwait++;
    if(d_priv->nwait == n){
	// Wake everybody up...
	d_priv->nwait=0;
	d_priv->cc=1-d_priv->cc;
	cond.conditionBroadcast();
    } else {
	cond.wait(d_priv->mutex);
} // End namespace SCIRun
    d_priv->mutex.unlock();
    Thread::couldBlockDone(oldstate);

