
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

#ifndef SCICore_Thread_RecursiveMutex_h
#define SCICore_Thread_RecursiveMutex_h

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
 
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/Mutex.h>

namespace SCICore {
    namespace Thread {
	class Thread;
	class RecursiveMutex_private;

	class RecursiveMutex {
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
	    const char* d_name;
	    RecursiveMutex_private* d_priv;

	    // Cannot copy them
	    RecursiveMutex(const RecursiveMutex&);
	    RecursiveMutex& operator=(const RecursiveMutex&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.6  1999/08/28 03:46:49  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:49  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:59  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

