
/*
 *  ThreadGroup: A set of threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_ThreadGroup_h
#define Core_Thread_ThreadGroup_h

#include <Core/share/share.h>

#include <Core/Thread/Mutex.h>
#include <vector>

namespace SCIRun {

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
	    Mutex lock_;
	    const char* name_;
	    ThreadGroup* parent_;
	    std::vector<ThreadGroup*> groups_;
	    std::vector<Thread*> threads_;
	    void addme(ThreadGroup* t);
	    void addme(Thread* t);

	    // Cannot copy them
	    ThreadGroup(const ThreadGroup&);
	    ThreadGroup& operator=(const ThreadGroup&);
	};
} // End namespace SCIRun

#endif


