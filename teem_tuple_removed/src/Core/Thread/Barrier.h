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

#ifndef Core_Thread_Barrier_h
#define Core_Thread_Barrier_h

#include <Core/share/share.h>

namespace SCIRun {

class Barrier_private;

/**************************************
 
 CLASS
 Barrier

 KEYWORDS
 Thread

 DESCRIPTION
 Barrier synchronization primitive.  Provides a single wait
 method to allow a set of threads to block at the barrier until all
 threads arrive.

 WARNING
 When the ThreadGroup semantics are used, other threads outside of the
 ThreadGroup should not access the barrier, or undefined behavior will
 result. In addition, threads should not be added or removed from the
 ThreadGroup while the Barrier is being accessed.
   
****************************************/

class SCICORESHARE Barrier {
public:
  //////////
  // Create a barrier which will be used by a variable number
  // of threads.   <tt>name</tt> should be a static string
  // which describes the primitive for debugging purposes.
  Barrier(const char* name);
    
  //////////
  // Destroy the barrier
  virtual ~Barrier();
    
  //////////
  // This causes all of the threads to block at this method
  // until all numThreads threads have called the method.
  // After all threads have arrived, they are all allowed
  // to return.
  void wait(int numThreads);

protected:
private:
  Barrier_private* priv_;
  const char* name_;

  // Cannot copy them
  Barrier(const Barrier&);
  Barrier& operator=(const Barrier&);
};
} // End namespace SCIRun

#endif

