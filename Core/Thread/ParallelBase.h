
/*
 *  ParallelBase: Helper class to instantiate several threads
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

#ifndef SCICore_Thread_ParallelBase_h
#define SCICore_Thread_ParallelBase_h

namespace SCICore {
    namespace Thread {
/**************************************
 
CLASS
   ParallelBase
   
KEYWORDS
   Thread
   
DESCRIPTION
   Helper class for Parallel class.  This will never be used
   by a user program.  See <b>Parallel</b> instead.
   
****************************************/
	class ParallelBase {
	public:
	    //////////
	    // <i>The thread body</i>
	    virtual void run(int proc)=0;

	protected:
	    ParallelBase();
	    virtual ~ParallelBase();
	    friend class Thread;

	private:
	    // Cannot copy them
	    ParallelBase(const ParallelBase&);
	    ParallelBase& operator=(const ParallelBase&);
	};
    }
}

#endif


//
// $Log$
// Revision 1.7  1999/09/02 16:52:43  sparker
// Updates to cocoon documentation
//
// Revision 1.6  1999/08/28 03:46:49  sparker
// Final updates before integration with PSE
//
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

