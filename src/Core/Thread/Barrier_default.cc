
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

using SCIRun::Barrier_private;
using SCIRun::Barrier;

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
    : name_(name)
{
    priv_=new Barrier_private;
}

Barrier::~Barrier()
{
    delete priv_;
}

void
Barrier::wait(int n)
{
    int oldstate=Thread::couldBlock(name_);
    priv_->mutex.lock();
    ConditionVariable& cond=priv_->cc?priv_->cond0:priv_->cond1;
    priv_->nwait++;
    if(priv_->nwait == n){
	// Wake everybody up...
	priv_->nwait=0;
	priv_->cc=1-priv_->cc;
	cond.conditionBroadcast();
    } else {
	cond.wait(priv_->mutex);
    }
    priv_->mutex.unlock();
    Thread::couldBlockDone(oldstate);
}
