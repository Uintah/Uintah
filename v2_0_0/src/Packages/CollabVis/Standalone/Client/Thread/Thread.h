/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Thread: The thread class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Thread_h
#define Core_Thread_Thread_h

#include <Core/Thread/Parallel.h>
#include <Core/Thread/Parallel1.h>
#include <Core/Thread/Parallel2.h>
#include <Core/Thread/Parallel3.h>
#include <Core/share/share.h>

namespace SCIRun {
	struct Thread_private;
	class ParallelBase;
	class Runnable;
	class ThreadGroup;
	
/**************************************
 
CLASS
   Thread
   
KEYWORDS
   Thread
   
DESCRIPTION

   The Thread class provides a new context in which to run.  A single
   Runnable class is attached to a single Thread class, which are
   executed in another thread.
   
****************************************/
	class SCICORESHARE Thread {
	public:
	    //////////
	    // Possible thread start states
	    enum ActiveState {
		Activated,
		Stopped,
		NotActivated
	    };
	    
	    //////////
	    // Create a thread, which will execute the <b>run()</b>
	    // method in the <b>runner</b> object. The thread <b>name</b>
	    // is used for identification purposes, and does not need to
	    // be unique with respect to other threads.  <b>Group</b>
	    // specifies the ThreadGroup that to which this thread
	    // should belong.  If no group is specified (group==0),
	    // the default group is used.
	    Thread(Runnable* runner, const char* name,
		   ThreadGroup* group=0, ActiveState state=Activated);
	    
	    //////////
	    // Return the <b>ThreadGroup</b> associated with this thread.
	    ThreadGroup* getThreadGroup();
	    
	    //////////
	    // Return the <b>Runnable</b> associated with this thread.
	    Runnable* getRunnable();
	    
	    //////////
	    // Flag the thread as a daemon thread.  When all non-deamon
	    // threads exit, the program will exit.
	    void setDaemon(bool to=true);
	    
	    //////////
	    // Returns true if the thread is tagged as a daemon thread.
	    bool isDaemon() const;
	    
	    //////////
	    // If the thread is started in the the NotActivated
	    // state, use this to activate the thread.
	    void activate(bool stopped);
	    
	    //////////
	    // Arrange to have the thread deleted automatically at exit.
	    // The pointer to the thread should not be used by any other
	    // threads once this has been called.
	    void detach();
	    
	    //////////
	    // Returns true if the thread is detached
	    bool isDetached() const;
	    
	    //////////
	    // Set the stack size for a particular thread.  In order
	    // to use this thread, you must create the thread in the
	    // stopped state, set the stack size, and then start the
	    // thread.  Setting the stack size for a thread that is
	    // running or has ever been run, will throw an exception.
	    void setStackSize(unsigned long stackSize);
	    
	    //////////
	    // Returns the stack size for the thread
	    unsigned long getStackSize() const;
	    
	    //////////
	    // Kill all threads and exit with <b>code</b>.
	    static void exitAll(int code);
	    
	    //////////
	    // Exit the currently running thread
	    static void exit();
	    
	    //////////
	    // Returns a pointer to the currently running thread.
	    static Thread* self();
	    
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
	    const char* getThreadName() const;
	    
	    //////////
	    // Returns the number of processors on the system
	    static int numProcessors();
	    
	    //////////
	    // Request that the thread migrate to processor <i>proc</i>.
	    // If <i>proc</i> is -1, then the thread is free to run
	    // anywhere.
	    void migrate(int proc);
	    
	    //////////
	    // Start up several threads that will run in parallel.  A new
	    // <b>ThreadGroup</b> is created as a child of the optional parent.
	    // If <i>block</i> is true, then the caller will block until all
	    // of the threads return.  Otherwise, the call will return
	    // immediately.
	    static ThreadGroup* parallel(const ParallelBase& helper,
					 int nthreads, bool block,
					 ThreadGroup* threadGroup=0);

	    //////////
	    // Start up several threads that will run in parallel.
	    // If <i>block</i> is true, then the caller will block until all
	    // of the threads return.  Otherwise, the call will return
	    // immediately.
	    template<class T>
	    static void parallel(T* ptr, void (T::*pmf)(int),
				 int numThreads, bool block) {
		parallel(Parallel<T>(ptr, pmf),
			 numThreads, block);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 1 argument
	    template<class T, class Arg1>
	    static void parallel(T* ptr, void (T::*pmf)(int, Arg1),
				 int numThreads, bool block,
				 Arg1 a1) {
		parallel(Parallel1<T, Arg1>(ptr, pmf, a1),
			 numThreads, block);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 2 arguments
	    template<class T, class Arg1, class Arg2>
	    static void parallel(T* ptr, void (T::* pmf)(int, Arg1, Arg2),
				 int numThreads, bool block,
				 Arg1 a1, Arg2 a2) {
		parallel(Parallel2<T, Arg1, Arg2>(ptr, pmf, a1, a2),
			 numThreads, block);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 3 arguments
	    template<class T, class Arg1, class Arg2, class Arg3>
	    static void parallel(T* ptr, void (T::* pmf)(int, Arg1, Arg2, Arg3),
				 int numThreads, bool block,
				 Arg1 a1, Arg2 a2, Arg3 a3) {
		parallel(Parallel3<T, Arg1, Arg2, Arg3>(ptr, pmf, a1, a2, a3),
			 numThreads, block);
	    }

	    //////////
	    // Abort the current thread, or the process.  Prints a message on
	    // stderr, and the user may choose one of:
	    // <pre>continue(c)/dbx(d)/cvd(v)/kill thread(k)/exit(e)</pre>
	    static void niceAbort();
	    
	    //////////
	    // Mark a section as one that could block for debugging purposes.
	    // The <b>int</b> that is returned should be passed into
	    // <i>couldBlockDone(int)</i> when the section has completed.  This
	    // will typically not be used outside of the thread implementation.
	    static int couldBlock(const char* why);
	    
	    //////////
	    // Mark the end of a selection that could block.
	    // <i>restore</i> was returned from a previous invocation
	    // of the above <b>couldBlock</b>.
	    static void couldBlockDone(int restore);
	    
	    //////////
	    // The calling process voluntarily gives up time to another process
	    static void yield();

	    //////////
	    // Return true if the thread library has been initialized. This
	    // will typically not be used outside of the thread implementation.
	    static bool isInitialized();

	    //////////
	    // SGI (irix 6.2-6.5.6 at least) maps page 0 for some
	    // OpenGL registers.  This is extremely silly, because now
	    // all programs that dereference null will not crash at
	    // the deref, and will be much harder to debug.  The
	    // thread library mprotects page 0 so that a deref of null
	    // WILL crash.  However, OpenGL programs then break.  This
	    // call unprotects page 0, making OpenGL work and also
	    // making a deref of 0 succeed.  You should call it before
	    // calling your first OpenGL function - usually
	    // glXQueryExtension or glXChooseVisual, glXGetConfig or
	    // similar.  Calling it multiple times is unncessary, but
	    // harmless.
	    static void allow_sgi_OpenGL_page0_sillyness();
	    static void disallow_sgi_OpenGL_page0_sillyness();
	private:
	    friend class Runnable;	    
	    friend class Thread_private;
	  // ejl
	  friend class ConditionVariable;
	  // /ejl
	    Runnable* runner_;
	    const char* threadname_;
	    ThreadGroup* group_;
	    unsigned long stacksize_;
	    bool daemon_;
	    bool detached_;
	    bool activated_;
	    void os_start(bool stopped);
	    Thread(ThreadGroup* g, const char* name);

	    static bool initialized;
	    static void initialize();
	    static void checkExit();
	    int cpu_;
	    ~Thread();
	    Thread_private* priv_;
	    static int id();
	    void run_body();	    	    
	    enum ThreadState {
		STARTUP,
		RUNNING,
		IDLE,
		SHUTDOWN,
		DIED,
		PROGRAM_EXIT,
		JOINING,
		BLOCK_ANY,
		BLOCK_BARRIER,
		BLOCK_MUTEX,
		BLOCK_SEMAPHORE,
		BLOCK_CONDITIONVARIABLE
	    };

	    static const char* getStateString(ThreadState);
	    static int push_bstack(Thread_private*, ThreadState s,
				   const char* why);
	    static void pop_bstack(Thread_private*, int oldstate);
	    int get_thread_id();
	    static void print_threads();

	    // Cannot copy them
	    Thread(const Thread&);
	    Thread& operator=(const Thread&);
	};	
} // End namespace SCIRun

#endif


