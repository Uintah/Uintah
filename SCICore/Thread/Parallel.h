
#ifndef SCI_THREAD_PARALLEL_H
#define SCI_THREAD_PARALLEL_H 1

/**************************************
 
CLASS
   Parallel
   
KEYWORDS
   Parallel
   
DESCRIPTION
   Helper class to make instantiating threads to perform a parallel
   task easier.
PATTERNS


WARNING
   
****************************************/

#include "ParallelBase.h"

template<class T>
class Parallel  : public ParallelBase {
    T* d_obj;
    void (T::*d_pmf)(int);
protected:
    virtual void run(int proc);
public:
    //////////
    // Create the parallel object similar to the above, but using the
    // specified member function instead of <i>parallel</i>.  This will
    // typically be used like:
    // <pre>Thread::parallel(Parallel&ltMyClass> (this, &ampMyClass::mymemberfn), nthreads)</pre>
    Parallel(T* obj, void (T::*pmf)(int));
    
    //////////
    // Destroy the Parallel object - the threads will remain alive.
    ~Parallel();
};

template<class T>
void
Parallel<T>::run(int proc)
{
    (d_obj->*d_pmf)(proc);
}

template<class T>
Parallel<T>::Parallel(T* obj, void (T::*pmf)(int))
    : d_obj(obj), d_pmf(pmf)
{
}

template<class T>
Parallel< T>::~Parallel()
{
}

#endif
