
// $Id$

/*
 *  Reducer: A barrier with reduction operations
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_Reducer_h
#define SCICore_Thread_Reducer_h

/**************************************
 
CLASS
   Reducer
   
KEYWORDS
   Thread
   
DESCRIPTION
   Perform reduction operations over a set of threads.  Reduction
   operations include things like global sums, global min/max, etc.
   In these operations, a local sum (operation) is performed on each
   thread, and these sums are added together.
 
 
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/Barrier.h>

namespace SCICore {
    namespace Thread {
	class ThreadGroup;

	class Reducer : public Barrier {
	public:
	    //////////
	    // Create a <b> Reducer</i> for the specified number of threads.
	    // At each operation, a barrier wait is performed, and the
	    // operation will be performed to compute the global balue.
	    // <i>name</i> should be a static string which describes
	    // the primitive for debugging purposes.
	    Reducer(const char* name, int nthreads);

	    //////////
	    // Create a <b>Reducer</b> to be associated with a particular
	    // <b>ThreadGroup</b>.
	    Reducer(const char* name, ThreadGroup* group);

	    //////////
	    // Destroy the reducer and free associated memory.
	    virtual ~Reducer();

	    //////////
	    // Performs a global sum over all of the threads.  As soon as each
	    // thread has called sum with their local sum, each thread will
	    // return the same global sum.
	    double sum(int proc, double mysum);

	    //////////
	    // Performs a global max over all of the threads.  As soon as each
	    // thread has called max with their local max, each thread will
	    // return the same global max.
	    double max(int proc, double mymax);

	private:
	    struct data {
		double d_d;
	    };
	    struct joinArray {
		data d_d;
		// Assumes 128 bytes in a cache line...
		char d_filler[128-sizeof(data)];
	    };
	    struct pdata {
		int d_buf;
		char d_filler[128-sizeof(int)];	
	    };
	    joinArray* d_join[2];
	    pdata* d_p;
	    int d_array_size;
	    void collectiveResize(int proc);
	};
    }
}

#endif

//
// $Log$
// Revision 1.5  1999/08/25 19:00:50  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:59  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

