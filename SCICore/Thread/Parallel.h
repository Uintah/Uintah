
// $Id$

/*
 *  Parallel: Automatically instantiate several threads
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

/**************************************
 
CLASS
   Parallel
   
KEYWORDS
   Thread
   
DESCRIPTION
   Helper class to make instantiating threads to perform a parallel
   task easier.
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/ParallelBase.h>

namespace SCICore {
    namespace Thread {
	template<class T> class Parallel  : public ParallelBase {
	public:
	    //////////
	    // Create a parallel object, using the specified member
	    // function instead of <i>parallel</i>.  This will
	    // typically be used like:
	    // <pre>Thread::parallel(Parallel&ltMyClass> (this, &ampMyClass::mymemberfn), nthreads)</pre>
	    Parallel(T* obj, void (T::*pmf)(int));
	    
	    //////////
	    // Destroy the Parallel object - the threads will remain alive.
	    virtual ~Parallel();
	    T* d_obj;
	    void (T::*d_pmf)(int);
	protected:
	    virtual void run(int proc);
	};
    }
}

template<class T>
void
SCICore::Thread::Parallel<T>::run(int proc)
{
    (d_obj->*d_pmf)(proc);
}

template<class T>
SCICore::Thread::Parallel<T>::Parallel(T* obj, void (T::*pmf)(int))
    : d_obj(obj), d_pmf(pmf)
{
}

template<class T>
SCICore::Thread::Parallel<T>::~Parallel()
{
}

#endif
//
// $Log$
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

