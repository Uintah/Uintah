
/*
 *  SimpleReducer: A barrier with reduction operations
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

#ifndef SCICore_Thread_SimpleReducer_h
#define SCICore_Thread_SimpleReducer_h

#include <SCICore/share/share.h>

#include <SCICore/Thread/Barrier.h>

namespace SCICore {
    namespace Thread {
	class ThreadGroup;

/**************************************
 
CLASS
   SimpleReducer
   
KEYWORDS
   Thread
   
DESCRIPTION
   Perform reduction operations over a set of threads.  Reduction
   operations include things like global sums, global min/max, etc.
   In these operations, a local sum (operation) is performed on each
   thread, and these sums are added together.
   
****************************************/
	class SCICORESHARE SimpleReducer : public Barrier {
	public:
	    //////////
	    // Create a <b> SimpleReducer</i>.
	    // At each operation, a barrier wait is performed, and the
	    // operation will be performed to compute the global balue.
	    // <i>name</i> should be a static string which describes
	    // the primitive for debugging purposes.
	    SimpleReducer(const char* name);

	    //////////
	    // Destroy the SimpleReducer and free associated memory.
	    virtual ~SimpleReducer();

	    //////////
	    // Performs a global sum over all of the threads.  As soon as each
	    // thread has called sum with their local sum, each thread will
	    // return the same global sum.
	    double sum(int myrank, int numThreads, double mysum);

	    //////////
	    // Performs a global max over all of the threads.  As soon as each
	    // thread has called max with their local max, each thread will
	    // return the same global max.
	    double max(int myrank, int numThreads, double mymax);

	    //////////
	    // Performs a global min over all of the threads.  As soon as each
	    // thread has called min with their local max, each thread will
	    // return the same global max.
	    double min(int myrank, int numThreads, double mymax);

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
	    void collectiveResize(int proc, int numThreads);

	    // Cannot copy them
	    SimpleReducer(const SimpleReducer&);
	    SimpleReducer& operator=(const SimpleReducer&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.5  2000/02/15 00:23:49  sparker
// Added:
//  - new Thread::parallel method using member template syntax
//  - Parallel2 and Parallel3 helper classes for above
//  - min() reduction to SimpleReducer
//  - ThreadPool class to help manage a set of threads
//  - unmap page0 so that programs will crash when they deref 0x0.  This
//    breaks OpenGL programs, so any OpenGL program linked with this
//    library must call Thread::allow_sgi_OpenGL_page0_sillyness()
//    before calling any glX functions.
//  - Do not trap signals if running within CVD (if DEBUGGER_SHELL env var set)
//  - Added "volatile" to fetchop barrier implementation to workaround
//    SGI optimizer bug
//
// Revision 1.4  1999/09/24 18:55:07  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.3  1999/09/02 16:52:44  sparker
// Updates to cocoon documentation
//
// Revision 1.2  1999/08/29 00:47:01  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.1  1999/08/28 03:46:50  sparker
// Final updates before integration with PSE
//
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

