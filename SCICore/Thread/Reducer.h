
#ifndef SCI_THREAD_REDUCER_H
#define SCI_THREAD_REDUCER_H 1

/**************************************
 
CLASS
   Reducer
   
KEYWORDS
   Reducer
   
DESCRIPTION
   Perform reduction operations over a set of threads.  Reduction
   operations include things like global sums, global min/max, etc.
   In these operations, a local sum (operation) is performed on each
   thread, and these sums are added together.
 
 
PATTERNS


WARNING
   
****************************************/

class ThreadGroup;
#include "Barrier.h"

class Reducer : public Barrier
{
    struct data {
	double d_d;
    };
    struct joinArray {
	data d_d;
	// Assumes 128 bytes in a cache line...
	char d_filler[128-sizeof(data)];
    };
    struct pdata {
	int d_buf;
	char d_filler[128-sizeof(int)];	
    };
    joinArray* d_join[2];
    pdata* d_p;
    int d_array_size;
    void collectiveResize(int proc);
public:
    //////////
    // Create a <b> Reducer</i> for the specified number of threads.  At
    // each operation, a barrier wait is performed, and the operation will
    // be performed to compute the global balue.  <i>name</i> should be a
    // string which describes the primitive for debugging purposes.
    Reducer(const std::string& name, int nthreads);

    //////////
    // Create a <b>Reducer</b> to be associated with a particular
    // <b>ThreadGroup</b>.
    Reducer(const std::string& name, ThreadGroup* group);

    //////////
    // Destroy the reducer and free associated memory.
    virtual ~Reducer();

    //////////
    // Performs a global sum over all of the threads.  As soon as each
    // thread has called sum with their local sum, each thread will
    // return the same global sum.
    double sum(int proc, double mysum);

    //////////
    // Performs a global max over all of the threads.  As soon as each
    // thread has called max with their local max, each thread will
    // return the same global max.
    double max(int proc, double mymax);
};

#endif
