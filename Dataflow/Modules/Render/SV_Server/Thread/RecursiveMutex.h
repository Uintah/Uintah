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

#ifndef Core_Thread_RecursiveMutex_h
#define Core_Thread_RecursiveMutex_h

#include <Core/share/share.h>

#include <Core/Thread/Mutex.h>

namespace SCIRun {

class RecursiveMutex_private;
/**************************************

 CLASS
 RecursiveMutex

 KEYWORDS
 Thread

 DESCRIPTION
 Provides a recursive <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
 <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
 Nested calls to <b>lock()</b> by the same thread are acceptable,
 but must be matched with calls to <b>unlock()</b>.  This class
 may be less efficient that the <b>Mutex</b> class, and should not
 be used unless the recursive lock feature is really required.
 
****************************************/
class SCICORESHARE RecursiveMutex {
public:
  //////////
  // Create the Mutex.  The Mutex is allocated in the unlocked
  // state. <i>name</i> should be a static string which describe
  // the primitive for debugging purposes.
  RecursiveMutex(const char* name);

  //////////
  // Destroy the Mutex.  Destroying a Mutex in the locked state
  // has undefined results.
  ~RecursiveMutex();

  //////////
  // Acquire the Mutex.  This method will block until the Mutex
  // is acquired.
  void lock();

  //////////
  // Release the Mutex, unblocking any other threads that are
  // blocked waiting for the Mutex.
  void unlock();

private:
  const char* name_;
  RecursiveMutex_private* priv_;

  // Cannot copy them
  RecursiveMutex(const RecursiveMutex&);
  RecursiveMutex& operator=(const RecursiveMutex&);
};
} // End namespace SCIRun

#endif


