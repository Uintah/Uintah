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
 *  Reducer: A barrier with reduction operations
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Reducer_h
#define Core_Thread_Reducer_h

#include <Core/Thread/Barrier.h>

namespace SCIRun {

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
 
****************************************/
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
		T data_;
		// Assumes 128 bytes in a cache line...
		char filler_[128];
	    };
	    DataArray* join_[2];

	    struct BufArray {
		int which;
		char filler_[128-sizeof(int)];
	    };
	    BufArray* p_;

	    int array_size_;
	    void collectiveResize(int proc, int numThreads);
	    void allocate(int size);

	    // Cannot copy them
	    Reducer(const Reducer<T>&);
	    Reducer<T>& operator=(const Reducer<T>&);
	};
    }
}

template<class T>
Reducer<T>::Reducer(const char* name, ReductionOp op)
    : Barrier(name), f_op(op)
{
    array_size_=-1;
    p_=0;
}

template<class T>
void
Reducer<T>::allocate(int n)
{
    join_[0]=new DataArray[2*numThreads+2]-1;
    join_[1]=join_[0]+numThreads;
    p_=new BufArray[num_threads_+2]+1;
    for(int i=0;i<num_threads_;i++)
        p_[i].whichBuffer_=0;
    array_size_=n;
}

template<class T>
Reducer<T>::~Reducer()
{
    if(p_){
	delete[] (void*)(join_[0]-1);
	delete[] (void*)(p_-1);
    }
}

template<class T>
void
Reducer<T>::collectiveResize(int proc, int n)
{
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait(n);
    if(proc==0){
	delete[] (void*)(join_[0]-1);
	delete[] (void*)(p_-1);
	allocate(n);
	array_size_=n;
    }
    wait(n);
}

template<class T>
T
Reducer<T>::reduce(int proc, int n, const T& myresult)
{
    if(n != array_size_){
        collectiveResize(proc, n);
} // End namespace SCIRun
    if(n<=1)
	return myresult;

    int buf=p_[proc].whichBuffer_;
    p_[proc].whichBuffer_=1-buf;

    dataArray* j=join_[buf];
    j[proc].data_=myresult;
    wait(n);
    T red=j[0].data_;
    for(int i=1;i<n;i++)
        red=(*f_op)(red, j[i].data_);
    return red;

#endif


