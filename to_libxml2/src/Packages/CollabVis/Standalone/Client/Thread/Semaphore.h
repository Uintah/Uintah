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
 *  Semaphore: Basic semaphore primitive
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Semaphore_h
#define Core_Thread_Semaphore_h

#include <Core/share/share.h>

namespace SCIRun {

class Semaphore_private;

/**************************************
 
 CLASS
 Semaphore

 KEYWORDS
 Thread

 DESCRIPTION
 Counting semaphore synchronization primitive.  A semaphore provides
 atomic access to a special counter.  The <i>up</i> method is used
 to increment the counter, and the <i>down</i> method is used to
 decrement the counter.  If a thread tries to decrement the counter
 when the counter is zero, that thread will be blocked until another
 thread calls the <i>up</i> method.

****************************************/
class SCICORESHARE Semaphore {
public:
  //////////
  // Create the semaphore, and setup the initial <i>count.name</i>
  // should be a static string which describes the primitive for
  // debugging purposes.
  Semaphore(const char* name, int count);

  //////////
  // Destroy the semaphore
  ~Semaphore();

  //////////
  // Increment the semaphore count, unblocking up to <i>count</i>
  // threads that may be blocked in the <i>down</i> method.
  void up(int count=1);
    
  //////////
  // Decrement the semaphore count by <i>count</i>.  If the
  // count is zero, this thread will be blocked until another
  // thread calls the <i>up</i> method. The order in which
  // threads will be unblocked is not defined, but implementors
  // should give preference to those threads that have waited
  // the longest.
  void down(int count=1);

  //////////
  // Attempt to decrement the semaphore count by one, but will
  // never block. If the count was zero, <i>tryDown</i> will
  // return false. Otherwise, <i>tryDown</i> will return true.
  bool tryDown();

private:
  Semaphore_private* priv_;
  const char* name_;

  // Cannot copy them
  Semaphore(const Semaphore&);
  Semaphore& operator=(const Semaphore&);
};
} // End namespace SCIRun

#endif

