
/*
 *  Parallel2: Automatically instantiate several threads, with 2 arguments
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2997
 *
 *  Copyright (C) 2997 SCI Group
 */

#ifndef SCICore_Thread_Parallel2_h
#define SCICore_Thread_Parallel2_h

#include <SCICore/Thread/ParallelBase.h>
#include <SCICore/Thread/Semaphore.h>

namespace SCICore {
    namespace Thread {
/**************************************
 
CLASS
   Parallel2
   
KEYWORDS
   Thread
   
DESCRIPTION
   Helper class to make instantiating threads to perform a parallel
   task easier.
   
****************************************/
	template<class T, class Arg1, class Arg2> class Parallel2  : public ParallelBase {
	public:
	    //////////
	    // Create a Parallel2 object, using the specified member
	    // function instead of <i>Parallel2</i>.  This will
	    // typically be used like:
	    // <b><pre>Thread::Parallel2(Parallel2&lt;MyClass&gt;(this, &amp;MyClass::mymemberfn), nthreads);</pre></b>
	    Parallel2(T* obj, void (T::*pmf)(int, Arg1, Arg2), Arg1 a1, Arg2 a2);
	    
	    //////////
	    // Destroy the Parallel2 object - the threads will remain alive.
	    virtual ~Parallel2();
	    T* d_obj;
	    void (T::*d_pmf)(int, Arg1, Arg2);
	    Arg1 a1;
	    Arg2 a2;
	protected:
	    virtual void run(int proc);
	private:
	    // Cannot copy them
	    Parallel2(const Parallel2&);
	    Parallel2<T, Arg1, Arg2>& operator=(const Parallel2<T, Arg1, Arg2>&);
	};
    }
}

template<class T, class Arg1, class Arg2>
void
SCICore::Thread::Parallel2<T, Arg1, Arg2>::run(int proc)
{
    // Copy out do make sure that the call is atomic
    T* obj=d_obj;
    void (T::*pmf)(int, Arg1, Arg2) = d_pmf;
    if(d_wait)
	d_wait->up();
    (obj->*pmf)(proc, a1, a2);
    // Cannot do anything here, since the object may be deleted by the
    // time we return
}

template<class T, class Arg1, class Arg2>
SCICore::Thread::Parallel2<T, Arg1, Arg2>::Parallel2(T* obj,
						     void (T::*pmf)(int, Arg1, Arg2),
						     Arg1 a1, Arg2 a2)
    : d_obj(obj), d_pmf(pmf), a1(a1), a2(a2)
{
    d_wait=0; // This may be set by Thread::parallel
}

template<class T, class Arg1, class Arg2>
SCICore::Thread::Parallel2<T, Arg1, Arg2>::~Parallel2()
{
}

#endif

//
// $Log$
// Revision 1.1  2000/02/15 00:23:49  sparker
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

