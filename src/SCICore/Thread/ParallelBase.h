
// $Id$

/*
 *  ParallelBase: Helper class to instantiate several threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_ParallelBase_h
#define SCICore_Thread_ParallelBase_h

/**************************************
 
CLASS
   ParallelBase
   
KEYWORDS
   Thread
   
DESCRIPTION
   Helper class for Parallel class.  This will never be used
   by a user program.  See <b>Parallel</b> instead.
PATTERNS


WARNING
   
****************************************/

namespace SCICore {
    namespace Thread {
	class ParallelBase {
	public:
	    //////////
	    // <i>The thread body</i>
	    virtual void run(int proc)=0;

	protected:
	    ParallelBase();
	    virtual ~ParallelBase();
	    friend class Thread;
	};
    }
}

#endif


//
// $Log$
// Revision 1.5  1999/08/25 19:00:49  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:58  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

