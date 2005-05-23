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
 *  Mutex: Standard locking primitive
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Mutex_h
#define Core_Thread_Mutex_h

#include <Core/share/share.h>

namespace SCIRun {

class Mutex_private;

/**************************************

 CLASS
 Mutex

 KEYWORDS
 Thread

 DESCRIPTION
 Provides a simple <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
 <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
 This is not a recursive Mutex (See <b>RecursiveMutex</b>), and calling
 lock() in a nested call will result in an error or deadlock.

****************************************/
class SCICORESHARE Mutex {
public:
  //////////
  // Create the mutex.  The mutex is allocated in the unlocked
  // state. <i>name</i> should be a static string which describes
  // the primitive for debugging purposes.  
  Mutex(const char* name);

  //////////
  // Destroy the mutex.  Destroying the mutex in the locked state
  // has undefined results.
  ~Mutex();

  //////////
  // Acquire the Mutex.  This method will block until the mutex
  // is acquired.
  void lock();

  //////////
  // Attempt to acquire the Mutex without blocking.  Returns
  // true if the mutex was available and actually acquired.
  bool tryLock();

  //////////
  // Release the Mutex, unblocking any other threads that are
  // blocked waiting for the Mutex.
  void unlock();
private:
  Mutex_private* priv_;
  const char* name_;

  // Cannot copy them
  Mutex(const Mutex&);
  Mutex& operator=(const Mutex&);
};

} // End namespace SCIRun

#endif


