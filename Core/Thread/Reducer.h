
/*
 *  Reducer: A barrier with reduction operations
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

#ifndef SCICore_Thread_Reducer_h
#define SCICore_Thread_Reducer_h

/**************************************
 
CLASS
   Reducer
   
KEYWORDS
   Thread
   
DESCRIPTION
   Perform reduction operations over a set of threads.  Reduction
   operations include things like global sums, global min/max, etc.
   In these operations, a local sum (operation) is performed on each
   thread, and these sums are added together.
 
 
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/Barrier.h>

namespace SCICore {
    namespace Thread {
	class ThreadGroup;

	template<class T> class Reducer : public Barrier {
	public:
	    //////////
	    // The function that performs the reduction
	    typedef T (*ReductionOp)(const T&, const T&);

	    //////////
	    // Create a <b> Reducer</i>.
	    // At each operation, a barrier wait is performed, and the
	    // operation will be performed to compute the global balue.
	    // <i>name</i> should be a static string which describes
	    // the primitive for debugging purposes.
	    // op is a function which will compute a reduced value from
	    // a pair of values.  op should be associative and commutative,
	    // even up to floating point errors.
	    Reducer(const char* name, ReductionOp);

	    //////////
	    // Destroy the Reducer and free associated memory.
	    virtual ~Reducer();

	    //////////
	    // Performs a global reduction over all of the threads.  As
	    // soon as each thread has called reduce with their local value,
	    // each thread will return the same global reduced value.
	    T reduce(int myrank, int numThreads, const T& value);

	private:
	    T (*f_op)(const T&, const T&);
	    struct DataArray {
		// We want this on it's own cache line
		T d_data;
		// Assumes 128 bytes in a cache line...
		char d_filler[128];
	    };
	    DataArray* d_join[2];

	    struct BufArray {
		int which;
		char d_filler[128-sizeof(int)];
	    };
	    BufArray* d_p;

	    int d_array_size;
	    void collectiveResize(int proc, int numThreads);
	    void allocate(int size);

	    // Cannot copy them
	    Reducer(const Reducer<T>&);
	    Reducer<T>& operator=(const Reducer<T>&);
	};
    }
}

template<class T>
SCICore::Thread::Reducer<T>::Reducer(const char* name, ReductionOp op)
    : Barrier(name), f_op(op)
{
    d_array_size=-1;
    d_p=0;
}

template<class T>
void
SCICore::Thread::Reducer<T>::allocate(int n)
{
    d_join[0]=new DataArray[2*numThreads+2]-1;
    d_join[1]=d_join[0]+numThreads;
    d_p=new BufArray[d_num_threads+2]+1;
    for(int i=0;i<d_num_threads;i++)
        d_p[i].d_whichBuffer=0;
    d_array_size=n;
}

template<class T>
SCICore::Thread::Reducer<T>::~Reducer()
{
    if(d_p){
	delete[] d_join[0]-1;
	delete[] d_p-1;
    }
}

template<class T>
void
SCICore::Thread::Reducer<T>::collectiveResize(int proc, int n)
{
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait(n);
    if(proc==0){
	delete[] d_join[0]-1;
	delete[] d_p-1;
	allocate(n);
	d_array_size=n;
    }
    wait(n);
}

template<class T>
T
SCICore::Thread::Reducer<T>::reduce(int proc, int n, const T& myresult)
{
    if(n != d_array_size){
        collectiveResize(proc, n);
    }
    if(n<=1)
	return myresult;

    int buf=d_p[proc].d_whichBuffer;
    d_p[proc].d_whichBuffer=1-buf;

    dataArray* j=d_join[buf];
    j[proc].d_data=myresult;
    wait(n);
    T red=j[0].d_data;
    for(int i=1;i<n;i++)
        red=(*f_op)(red, j[i].d_data);
    return red;
}

#endif

//
// $Log$
// Revision 1.7  1999/08/29 00:47:01  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/28 03:46:49  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:50  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:59  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

