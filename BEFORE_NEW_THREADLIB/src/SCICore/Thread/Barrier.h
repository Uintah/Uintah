
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

#ifndef SCICore_Thread_Barrier_h
#define SCICore_Thread_Barrier_h

/**************************************
 
CLASS
   Barrier
   
KEYWORDS
   Thread
   
DESCRIPTION
   Barrier synchronization primitive.  Provides a single wait
   method to allow a set of threads to block at the barrier until all
   threads arrive.
PATTERNS


WARNING
   When the ThreadGroup semantics are used, other threads outside of the
   ThreadGroup should not access the barrier, or undefined behavior will
   result. In addition, threads should not be added or removed from the
   ThreadGroup while the Barrier is being accessed.
   
****************************************/

namespace SCICore {
    namespace Thread {
	class ThreadGroup;
	class Barrier_private;

	class Barrier {
	public:
	    //////////
	    // Create a barrier which will be used by nthreads threads.
	    // <tt>name</tt> should be a static string which describes the
	    // primitive for debugging purposes.
	    Barrier(const char* name, int numThreads);
    
	    //////////
	    // Create a Barrier to be associated with a particular
	    // ThreadGroup.
	    Barrier(const char* name, ThreadGroup* group);
    
	    //////////
	    // Destroy the barrier
	    virtual ~Barrier();
    
	    //////////
	    // This causes all of the threads to block at this method
	    // until all numThreads threads have called the method.
	    // After all threads have arrived, they are all allowed
	    // to return.
	    void wait();

	protected:
	    int d_num_threads;
	    ThreadGroup* d_thread_group;

	private:
	    Barrier_private* d_priv;
	    const char* d_name;

	    // Cannot copy them
	    Barrier(const Barrier&);
	    Barrier& operator=(const Barrier&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.6  1999/08/28 03:46:46  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:46  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:54  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
