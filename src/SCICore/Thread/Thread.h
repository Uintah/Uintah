
// $Id$

/*
 *  Thread.h: The thread class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_Thread_h
#define SCICore_Thread_Thread_h


/**************************************
 
CLASS
   Thread
   
KEYWORDS
   Thread
   
DESCRIPTION
 
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/ParallelBase.h>
#include <SCICore/Thread/Runnable.h>

namespace SCICore {
    namespace Thread {
	struct Thread_private;
	class ThreadGroup;
	
	class Thread {
	    friend class Mutex;
	    friend class Semaphore;
	    friend class Barrier;
	    
	    Thread(const Thread&);
	    
	    ~Thread();
	    bool d_daemon;
	    bool d_detached;
	    Runnable* d_runner;
	    const std::string d_threadname;
	    Thread_private* d_priv;
	    ThreadGroup* d_group;
	    int d_cpu;
	    int d_priority;
	    
	    //////////
	    // This method is specific to a particular thread implementation.
	    void os_start(bool stopped);
	    
	    //////////
	    // This is for an internal initialization.  Do not call it
	    // directly.
	    static void initialize();
	    
	    //////////
	    // Private constructor for internal use only
	    Thread(ThreadGroup* g, const std::string& name);
	    friend class Runnable;
	    
	    friend void Thread_run(Thread*);
	    
	    friend void Thread_shutdown(Thread*);
	    
	    void run_body();
	    class ParallelHelper : public Runnable {
		const ParallelBase* d_helper;
		int d_proc;
	    public:
		ParallelHelper(const ParallelBase* helper, int proc)
		    : d_helper(helper), d_proc(proc) {}
		virtual ~ParallelHelper() {}
		virtual void run() {
		    ParallelBase* cheat=(ParallelBase*)d_helper;
		    cheat->run(d_proc);
		}
	    };
	    
	    static void checkExit();
	public:
	    //////////
	    // Create a thread, which will execute the <b>run()</b>
	    // method in the <b>runner</b> object. The thread <b>name</b>
	    // is used for identification purposes, and does not need to
	    // be unique with respect to other threads.  <b>Group</b>
	    // specifies the ThreadGroup that to which this thread
	    // should belong.  If no group is specified (group==0),
	    // the default group is used.
	    Thread(Runnable* runner, const std::string& name,
		   ThreadGroup* group=0, bool stopped=false);
	    
	    //////////
	    // Return the <b>ThreadGroup</b> associated with this thread.
	    ThreadGroup* threadGroup();
	    
	    //////////
	    // Flag the thread as a daemon thread.  When all non-deamon
	    // threads exit, the program will exit.
	    void setDaemon(bool to=true);
	    
	    //////////
	    // Set the priority for the thread.  Priorities range from
	    // 1 to 10, with 10 having the highest priority.  The default
	    // priority is 5.
	    void setPriority(int priority);
	    
	    //////////
	    // Return the current priority of the thread.
	    int getPriority() const;
	    
	    //////////
	    // Returns true if the thread is tagged as a daemon thread.
	    bool isDaemon() const;
	    
	    //////////
	    // Arrange to have the thread deleted automatically at exit.
	    // The pointer to the thread should not be used by any other
	    // threads once this has been called.
	    void detach();
	    
	    //////////
	    // Returns true if the thread is tagged as a daemon thread
	    bool isDetached() const;
	    
	    //////////
	    // Kill all threads and exit with <b>code</b>.
	    static void exitAll(int code);
	    
	    //////////
	    // Returns a pointer to the currently running thread.
	    static Thread* currentThread();
	    
	    //////////
	    // Stop the thread.
	    void stop();
	    
	    //////////
	    // Resume the thread
	    void resume();
	    
	    //////////
	    // Blocks the calling thread until this thead has finished
	    // executing. You cannot join detached threads or daemon threads.
	    void join();
	    
	    //////////
	    // Returns the name of the thread
	    const std::string& threadName() const;
	    
	    //////////
	    // Returns the number of processors on the system
	    static int numProcessors();
	    
	    //////////
	    // Request that the thread migrate to processor <i>proc</i>.
	    // If <i>proc</i> is -1, then the thread is free to run
	    // anywhere.
	    void migrate(int proc);
	    
	    //////////
	    // Returns the private pointer - should only be used by
	    // the thread implementation
	    Thread_private* getPrivate() const;
	    
	    //////////
	    // Start up several threads that will run in parallel.  A new
	    // <b>ThreadGroup</b> is created as a child of the optional parent.
	    // If <i>block</i> is true, theen the caller will block until all
	    // of the threads return.  Otherwise, the call will return
	    // immediately.
	    static ThreadGroup* parallel(const ParallelBase& helper,
					 int nthreads, bool block=false,
					 ThreadGroup* threadGroup=0);
	    
	    //////////
	    // Abort the current thread, or the process.  Prints a message on
	    // stderr, and the user may choose one of:
	    // <pre>continue(c)/dbx(d)/cvd(v)/kill thread(k)/exit(e)</pre>
	    static void niceAbort();
	    
	    //////////
	    // Mark a section as one that could block for debugging purposes.
	    // The <b>int</b> that is returned should be passed into
	    // <i>couldBlock(int)</i> when the section has completed.  This
	    // will typically not be used outside of the thread implementation
	    static int couldBlock(const char* why);
	    
	    //////////
	    // Mark the end of a selection that could block.
	    // <i>restore</i> was returned from a previous invocation
	    // of the above <b>couldBlock</b>.
	    static void couldBlockDone(int restore);
	    
	    //////////
	    // Voluntarily give up time to another processor
	    static void yield();
	    
	};	
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/08/25 02:38:00  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

