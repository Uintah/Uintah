
// $Id$

/*
 *  Guard.h: Automatically lock/unlock a mutex or crowdmonitor.
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

/**************************************
 
CLASS
   Guard
   
KEYWORDS
   Guard
   
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
PATTERNS


WARNING
   
****************************************/

class SCICore {
    class Thread {
	class CrowdMonitor;
	class Mutex;

	class Guard {
	    Mutex* d_mutex;
	    CrowdMonitor* d_monitor;
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
	    Which action;
	};
    }
}

#endif

//
// $Log$
// Revision 1.2  1999/08/25 02:37:56  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
