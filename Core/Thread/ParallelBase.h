
$Id$

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
   ParallelBase
   
DESCRIPTION
   Helper class for Parallel class.  This will never be used
   by a user program.  See <b>Parallel</b> instead.
PATTERNS


WARNING
   
****************************************/

namespace SCICore {
    namespace Thread {
	class ParallelBase {
	protected:
	    ParallelBase();
	    virtual ~ParallelBase();
	    friend class Thread;
	public:
	    //////////
	    // <i>The thread body</i>
	    virtual void run(int proc)=0;
	};
    }
}

#endif


//
// $Log$
// Revision 1.4  1999/08/25 02:37:58  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

