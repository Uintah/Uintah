
/*
 *  ThreadPool: A pool of threads
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCICore_Thread_ThreadPool_h
#define SCICore_Thread_ThreadPool_h

#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Parallel2.h>
#include <SCICore/Thread/Parallel3.h>
#include <SCICore/share/share.h>
#include <vector>

namespace SCICore {
    namespace Thread {
	class ParallelBase;
	class ThreadPoolHelper;
	class ThreadGroup;
	
/**************************************
 
CLASS
   ThreadPool
   
KEYWORDS
   ThreadPool
   
DESCRIPTION

   The ThreadPool class groups a bunch of worker threads.
   
****************************************/
	class SCICORESHARE ThreadPool {
	public:
	    //////////
	    // Create a thread pool.  <tt>name</tt> should be a static
	    // string which describes the primitive for debugging purposes.
	    ThreadPool(const char* name);

	    //////////
	    // Destroy the pool and shutdown all threads
	    ~ThreadPool();

	    //////////
	    // Start up several threads that will run in parallel.
	    // The caller will block until all of the threads return.
	    void parallel(const ParallelBase& helper, int nthreads);

	    //////////
	    // Start up several threads that will run in parallel.
	    // The caller will block until all of the threads return.
	    template<class T>
	    void parallel(T* ptr, void (T::*pmf)(int), int numThreads) {
		parallel(Parallel<T>(ptr, pmf),
			 numThreads);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 1 argument
	    template<class T, class Arg1>
	    void parallel(T* ptr, void (T::*)(int, Arg1),
			  int numThreads,
			  Arg1 a1) {
		parallel(Parallel1<T, Arg1>(ptr, pmf, a1),
			 numThreads);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 2 arguments
	    template<class T, class Arg1, class Arg2>
	    void parallel(T* ptr, void (T::* pmf)(int, Arg1, Arg2),
			  int numThreads,
			  Arg1 a1, Arg2 a2) {
		parallel(Parallel2<T, Arg1, Arg2>(ptr, pmf, a1, a2),
			 numThreads);
	    }

	    //////////
	    // Another overloaded version of parallel that passes 3 arguments
	    template<class T, class Arg1, class Arg2, class Arg3>
	    void parallel(T* ptr, void (T::* pmf)(int, Arg1, Arg2, Arg3),
			  int numThreads,
			  Arg1 a1, Arg2 a2, Arg3 a3) {
		parallel(Parallel3<T, Arg1, Arg2, Arg3>(ptr, pmf, a1, a2, a3),
			 numThreads);
	    }

	private:
	    const char* d_name;
	    ThreadGroup* d_group;
	    Mutex d_lock;
	    std::vector<ThreadPoolHelper*> d_threads;
	    int d_numThreads;
	    Barrier barrier;
	    friend class ThreadPoolHelper;
	    void wait();

	    // Cannot copy them
	    ThreadPool(const ThreadPool&);
	    ThreadPool& operator=(const ThreadPool&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  2000/02/15 00:23:50  sparker
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
//
