
/*
 *  Barrier: Barrier synchronization primitive
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

#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>

namespace SCICore {
    namespace Thread {
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

using SCICore::Thread::Barrier_private;
using SCICore::Thread::Barrier;

Barrier_private::Barrier_private()
    : mutex("Barrier lock"),
      cond0("Barrier condition 0"), cond1("Barrier condition 1"),
      cc(0), nwait(0)
{
}

Barrier_private::~Barrier_private()
{
}

Barrier::Barrier(const char* name, int num_threads)
    : d_num_threads(num_threads), d_thread_group(0), d_name(name)
{
    d_priv=new Barrier_private;
}

Barrier::Barrier(const char* name, ThreadGroup* thread_group)
 : d_num_threads(0), d_thread_group(thread_group), d_name(name)
{
    d_priv=new Barrier_private;
}

Barrier::~Barrier()
{
    delete d_priv;
}

void
Barrier::wait()
{
    int n=d_thread_group?d_thread_group->numActive(true):d_num_threads;
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
    }
    d_priv->mutex.unlock();
    Thread::couldBlockDone(oldstate);
}

//
// $Log$
// Revision 1.2  1999/08/28 03:46:46  sparker
// Final updates before integration with PSE
//
// Revision 1.1  1999/08/25 19:00:47  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
