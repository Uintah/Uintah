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
 *  ThreadPool: A pool of threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Core_Thread_ThreadPool_h
#define Core_Thread_ThreadPool_h

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Parallel2.h>
#include <Core/Thread/Parallel3.h>
#include <Core/share/share.h>
#include <vector>

namespace SCIRun {

class 	ThreadPoolHelper;
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
  const char* name_;
  ThreadGroup* group_;
  Mutex lock_;
  std::vector<ThreadPoolHelper*> threads_;
  int numThreads_;
  Barrier barrier;
  friend class ThreadPoolHelper;
  void wait();

  // Cannot copy them
  ThreadPool(const ThreadPool&);
  ThreadPool& operator=(const ThreadPool&);
};
} // End namespace SCIRun

#endif

