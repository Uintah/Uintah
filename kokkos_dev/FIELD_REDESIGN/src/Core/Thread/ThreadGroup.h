
/*
 *  ThreadGroup: A set of threads
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

#ifndef SCICore_Thread_ThreadGroup_h
#define SCICore_Thread_ThreadGroup_h

#include <SCICore/share/share.h>

#include <SCICore/Thread/Mutex.h>
#include <vector>

namespace SCICore {
    namespace Thread {
	class Thread;

/**************************************
 
CLASS
   ThreadGroup
   
KEYWORDS
   Thread
   
DESCRIPTION
   A group of threads that are linked together for scheduling
   and control purposes.  The threads may be stopped, resumed
   and alerted simultaneously.
 
****************************************/
	class SCICORESHARE ThreadGroup {
	public:
	    
	    //////////
	    // Create a thread group with the specified <i>name</i>.
	    // <i>parentGroup</i> specifies the parent <b>ThreadGroup</b>
	    // which defaults to the default top-level group.
	    ThreadGroup(const char* name, ThreadGroup* parentGroup=0);
	    
	    //////////
	    // Destroy the thread group.  All of the running threads
	    // should be stopped before the <b>ThreadGroup</b> is destroyed.
	    ~ThreadGroup();
	    
	    //////////
	    // Return a snapshot of the number of living threads.  If
	    // <i>countDaemon</i> is true, then daemon threads will be
	    // included in the count.
	    int numActive(bool countDaemon);
	    
	    //////////
	    // Stop all of the threads in this thread group
	    void stop();
	    
	    //////////
	    // Resume all of the threads in this thread group
	    void resume();
	    
	    //////////
	    // Wait until all of the threads have completed.
	    void join();
	    
	    //////////
	    // Detach the thread, joins are no longer possible.
	    void detach();
	    
	    //////////
	    // Return the parent <b>ThreadGroup.</b>  Returns null if
	    // this is the default threadgroup.
	    ThreadGroup* parentGroup();
	    
	    //////////
	    // Arrange to have the threadGroup gang scheduled, so that
	    // all of the threads will be executing at the same time if
	    // multiprocessing resources permit.  This interface will
	    // typically be employed by the <i>Thread::parallel</i>
	    // static method, and will typically not be called directly
	    // by user code.  Threads added to the group after this call
	    // may or may not be included in the schedule gang. 
	    void gangSchedule();

	protected:
	    friend class Thread;
	    static ThreadGroup* s_default_group;

	private:
	    Mutex d_lock;
	    const char* d_name;
	    ThreadGroup* d_parent;
	    std::vector<ThreadGroup*> d_groups;
	    std::vector<Thread*> d_threads;
	    void addme(ThreadGroup* t);
	    void addme(Thread* t);

	    // Cannot copy them
	    ThreadGroup(const ThreadGroup&);
	    ThreadGroup& operator=(const ThreadGroup&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.9  2000/02/15 00:23:50  sparker
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
// Revision 1.8  1999/09/24 18:55:08  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.7  1999/09/02 16:52:44  sparker
// Updates to cocoon documentation
//
// Revision 1.6  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:52  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:38:01  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

