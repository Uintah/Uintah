
#ifndef SCI_THREAD_BARRIER_H
#define SCI_THREAD_BARRIER_H

/**************************************
 
CLASS
   Barrier
   
KEYWORDS
   Barrier
   
DESCRIPTION
   Barrier synchronization primitive.  Provides a single wait
   method to allow a set of threads to block at the barrier until all
   threads arrive.
PATTERNS


WARNING
   When the ThreadGroup semantics are used, other threads outside of the
   ThreadGroup should not access the barrier, or undefined behavior will
   result. In addition, threads should not be added or removed from the
   ThreadGroup while the Barrier is being accessed.
   
****************************************/

class ThreadGroup;
class Barrier_private;
#include <string>

class Barrier {
    Barrier_private* d_priv;
    std::string d_name;
protected:
    int d_num_threads;
    ThreadGroup* d_thread_group;
public:
    //////////
    // Create a barrier which will be used by nthreads threads.
    // <tt>name</tt> should be a string which describes the
    // primitive for debugging purposes.
    Barrier(const std::string& name, int nthreads);
    
    //////////
    // Create a Barrier to be associated with a particular ThreadGroup.
    Barrier(const std::string& name, ThreadGroup* group);
    
    //////////
    // Destroy the barrier
    virtual ~Barrier();
    
    //////////
    // This causes all of the threads to block at this method until all
    // nthreads threads have called the method.  After all threads have
    // arrived, they are all allowed to return.
    void wait();
};

#endif

