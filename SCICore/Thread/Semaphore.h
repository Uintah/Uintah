
#ifndef SCI_THREAD_SEMAPHORE_H
#define SCI_THREAD_SEMAPHORE_H 1

/**************************************
 
CLASS
   Semaphore
   
KEYWORDS
   Semaphore
   
DESCRIPTION
   Counting semaphore synchronization primitive.  A semaphore provides
   atomic access to a special counter.  The <i>up</i> method is used
   to increment the counter, and the <i>down</i> method is used to
   decrement the counter.  If a thread tries to decrement the counter
   when the counter is zero, that thread will be blocked until another
   thread calls the <i>up</i> method.

PATTERNS


WARNING
   
****************************************/

class Semaphore_private;
#include <string>

class Semaphore {
    Semaphore_private* d_priv;
    const std::string d_name;
public:
    //////////
    // Create the semaphore, and setup the initial <i>count.name</i>
    // should be a string which describes the primitive for
    // debugging purposes.
    Semaphore(const std::string& name, int count);

    //////////
    // Destroy the semaphore
    ~Semaphore();

    //////////
    // Increment the semaphore count, unblocking up to one thread that may
    // be blocked in the <i>down</i> method.
    void up();
    
    //////////
    // Decrement the semaphore count.  If the count is zero, this thread
    // will be blocked until another thread calls the <i>up</i> method.
    // The order in which threads will be unblocked is not defined, but
    // implementors should give preference to those threads that have waited
    // the longest.
    void down();

    //////////
    // Attempt to decrement the semaphore count, but will never block.
    // If the count was zero, <i>tryDown</i> will return false.
    // Otherwise, <i>tryDown</i> will return true.
    bool tryDown();
};

#endif
