
/*
 *  MutexPool: A set of mutex objects
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Thread_MutexPool_h
#define SCICore_Thread_MutexPool_h

#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/share/share.h>

namespace SCICore {
    namespace Thread {
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
	    AtomicCounter d_nextID;

 	    //////////
	    // The number of Mutexes in the pool
	    int d_size;

 	    //////////
	    // The array of Mutex objects.
	    Mutex** d_pool;

 	    //////////
	    // Private copy ctor to prevent accidental copying
	    MutexPool(const MutexPool&);
 	    //////////
	    // Private assignment operator to prevent accidental assignment
	    MutexPool& operator=(const MutexPool&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  1999/09/25 08:29:29  sparker
// Added MutexPool class - a utility for sharing Mutexes among a large
//  number of objects
// Fixed comments in Guard
//
//
