
/*
 *  Parallel: Automatically instantiate several threads
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

#ifndef SCICore_Thread_Parallel_h
#define SCICore_Thread_Parallel_h

#include <SCICore/Thread/ParallelBase.h>
#include <SCICore/Thread/Semaphore.h>

namespace SCICore {
    namespace Thread {
/**************************************
 
CLASS
   Parallel
   
KEYWORDS
   Thread
   
DESCRIPTION
   Helper class to make instantiating threads to perform a parallel
   task easier.
   
****************************************/
	template<class T> class Parallel  : public ParallelBase {
	public:
	    //////////
	    // Create a parallel object, using the specified member
	    // function instead of <i>parallel</i>.  This will
	    // typically be used like:
	    // <b><pre>Thread::parallel(Parallel&lt;MyClass&gt;(this, &amp;MyClass::mymemberfn), nthreads);</pre></b>
	    Parallel(T* obj, void (T::*pmf)(int));
	    
	    //////////
	    // Destroy the Parallel object - the threads will remain alive.
	    virtual ~Parallel();
	    T* d_obj;
	    void (T::*d_pmf)(int);
	protected:
	    virtual void run(int proc);
	private:
	    // Cannot copy them
	    Parallel(const Parallel&);
	    Parallel<T>& operator=(const Parallel<T>&);
	};
    }
}

template<class T>
void
SCICore::Thread::Parallel<T>::run(int proc)
{
    // Copy out do make sure that the call is atomic
    T* obj=d_obj;
    void (T::*pmf)(int) = d_pmf;
    if(d_wait)
	d_wait->up();
    (obj->*pmf)(proc);
    // Cannot do anything here, since the object may be deleted by the
    // time we return
}

template<class T>
SCICore::Thread::Parallel<T>::Parallel(T* obj, void (T::*pmf)(int))
    : d_obj(obj), d_pmf(pmf)
{
    d_wait=0; // This may be set by Thread::parallel
}

template<class T>
SCICore::Thread::Parallel<T>::~Parallel()
{
}

#endif

//
// $Log$
// Revision 1.8  2000/02/15 00:23:49  sparker
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
// Revision 1.7  1999/09/03 19:51:15  sparker
// Fixed bug where if Thread::parallel was called with block=false, the
//   helper object could get destroyed before it was used.
// Removed include of SCICore/Thread/ParallelBase and
//  SCICore/Thread/Runnable from Thread.h to minimize dependencies
// Fixed naming of parallel helper threads.
//
// Revision 1.6  1999/09/02 16:52:43  sparker
// Updates to cocoon documentation
//
// Revision 1.5  1999/08/28 03:46:48  sparker
// Final updates before integration with PSE
//
// Revision 1.4  1999/08/25 19:00:49  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:57  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

