
/*
 *  Guard: Automatically lock/unlock a mutex or crowdmonitor.
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

#ifndef SCICore_Thread_Guard_h
#define SCICore_Thread_Guard_h

namespace SCICore {
    namespace Thread {
	class CrowdMonitor;
	class Mutex;

/**************************************
 
CLASS
   Guard
   
KEYWORDS
   Thread
   
DESCRIPTION
   Utility class to lock and unlock a <b>Mutex</b> or a <b>CrowdMonitor</b>.
   The constructor of the <b>Guard</b> object will lock the mutex
   (or <b>CrowdMonitor</b>), and the destructor will unlock it.
   <p>
   This would be used like this:
   <blockquote><pre>
   {
   <blockquote>Guard mlock(&mutex);  // Acquire the mutex
       ... critical section ...</blockquote>
   } // mutex is released when mlock goes out of scope
   </pre></blockquote>
   
****************************************/
	class Guard {
	public:
	    //////////
	    // Attach the <b>Guard</b> object to the <i>mutex</i>, and
	    // acquire the mutex.
	    Guard(Mutex* mutex);
	    enum Which {
		Read,
		Write
	    };
    
	    //////////
	    // Attach the <b>Guard</b> to the <i>CrowdMonitor</pre> and
	    // acquire one of the locks.  If <i>action</i> is
	    // <b>Guard::Read</b>, the read lock will be acquired, and if
	    // <i>action</i> is <b>Write</b>, then the write lock will be
	    // acquired.  The appropriate lock will then be released by the
	    // destructor
	    Guard(CrowdMonitor* crowdMonitor, Which action);
    
	    //////////
	    // Release the lock acquired by the constructor.
	    ~Guard();
	private:
	    Mutex* d_mutex;
	    CrowdMonitor* d_monitor;
	    Which d_action;

	    // Cannot copy them
	    Guard(const Guard&);
	    Guard& operator=(const Guard&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.5  1999/09/02 16:52:42  sparker
// Updates to cocoon documentation
//
// Revision 1.4  1999/08/28 03:46:48  sparker
// Final updates before integration with PSE
//
// Revision 1.3  1999/08/25 19:00:48  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.2  1999/08/25 02:37:56  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
