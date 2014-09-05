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
 *  MutexPool: A set of mutex objects
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Thread_MutexPool_h
#define Core_Thread_MutexPool_h

#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/share/share.h>

namespace SCIRun {
/**************************************
 
CLASS
   MutexPool
   
KEYWORDS
   Thread, Mutex
   
DESCRIPTION
   A container class for a set of Mutex objects.  This can be used to
   limit the number of active mutexes down to a more reasonable set.
   However, this must be used very carefully, as it becomes easy to
   create a hold-and-wait condition.
****************************************/
	class SCICORESHARE MutexPool {
	public:
	    //////////
	    // Create the mutex pool with size mutex objects.
	    MutexPool(const char* name, int size);

	    //////////
	    // Destroy the mutex pool and all mutexes in it.
	    ~MutexPool();

	    //////////
	    // return the next index in a round-robin fashion
	    int nextIndex();

	    //////////
	    // return the idx'th mutex.
	    Mutex* getMutex(int idx);

	    //////////
	    // lock the idx'th mutex.
	    void lockMutex(int idx);

	    //////////
	    // unlock the idx'th mutex.
	    void unlockMutex(int idx);
	private:
	    //////////
	    // The next ID
	    AtomicCounter nextID_;

 	    //////////
	    // The number of Mutexes in the pool
	    int size_;

 	    //////////
	    // The array of Mutex objects.
	    Mutex** pool_;

 	    //////////
	    // Private copy ctor to prevent accidental copying
	    MutexPool(const MutexPool&);
 	    //////////
	    // Private assignment operator to prevent accidental assignment
	    MutexPool& operator=(const MutexPool&);
	};
} // End namespace SCIRun

#endif

