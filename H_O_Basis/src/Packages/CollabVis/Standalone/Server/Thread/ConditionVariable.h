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

#ifndef Core_Thread_ConditionVariable_h
#define Core_Thread_ConditionVariable_h

#include <Core/share/share.h>

#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>

struct timespec;

namespace SCIRun {

class ConditionVariable_private;
class CrowdMonitor_private;

/**************************************
 
  CLASS
  ConditionVariable

  KEYWORDS
  Thread

  DESCRIPTION
  Condition variable primitive.  When a thread calls the
  <i>wait</i> method,which will block until another thread calls
  the <i>conditionSignal</i> or <i>conditionBroadcast</i> methods.  When
  there are multiple threads waiting, <i>conditionBroadcast</i> will unblock
  all of them, while <i>conditionSignal</i> will unblock only one (an
  arbitrary one) of them.  This primitive is used to allow a thread
  to wait for some condition to exist, such as an available resource.
  The thread waits for that condition, until it is unblocked by another
  thread that caused the condition to exist (<i>i.e.</i> freed the
  resource).
   
****************************************/

class SCICORESHARE ConditionVariable {
public:
  //////////
  // Create a condition variable. <i>name</i> should be a static
  // string which describes the primitive for debugging purposes.
  ConditionVariable(const char* name);
    
  //////////
  // Destroy the condition variable
  ~ConditionVariable();
    
  //////////
  // Wait for a condition.  This method atomically unlocks
  // <b>mutex</b>, and blocks.  The <b>mutex</b> is typically
  // used to guard access to the resource that the thread is
  // waiting for.
  void wait(Mutex& m);

  //////////
  // Wait for a condition.  This method atomically unlocks
  // <b>mutex</b>, and blocks.  The <b>mutex</b> is typically
  // used to guard access to the resource that the thread is
  // waiting for.  If the time abstime is reached before
  // the ConditionVariable is signaled, this will return
  // false.  Otherewise it will return true.
  bool timedWait(Mutex& m, const struct ::timespec* abstime);
    
  //////////
  // Signal a condition.  This will unblock one of the waiting
  // threads. No guarantee is made as to which thread will be
  // unblocked, but thread implementations typically give
  // preference to the thread that has waited the longest.
  void conditionSignal();

  //////////
  // Signal a condition.  This will unblock all of the waiting
  // threads. Note that only the number of waiting threads will
  // be unblocked. No guarantee is made that these are the same
  // N threads that were blocked at the time of the broadcast.
  void conditionBroadcast();

private:
  const char* name_;
  ConditionVariable_private* priv_;

  // Cannot copy them
  ConditionVariable(const ConditionVariable&);
  ConditionVariable& operator=(const ConditionVariable&);
};
} // End namespace SCIRun

#endif

