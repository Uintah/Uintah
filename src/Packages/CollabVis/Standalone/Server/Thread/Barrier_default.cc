/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "Barrier: %s\n", name);
    Thread::initialize();
  }
  priv_=new Barrier_private;
}

Barrier::~Barrier()
{
    delete priv_;
    priv_=0;
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
