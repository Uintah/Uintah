
#ifndef SCI_THREAD_MUTEX_H
#define SCI_THREAD_MUTEX_H 1

/**************************************
 
CLASS
   Mutex
   
KEYWORDS
   Mutex
   
DESCRIPTION
   Provides a simple <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
   <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
   This is not a recursive Mutex (See <b>RecursiveMutex</b>), and calling
   lock() in a nested call will result in an error or deadlock.
PATTERNS


WARNING
   
****************************************/

class Mutex_private;
#include <string>

class Mutex {
    Mutex_private* d_priv;
    std::string d_name;
public:
    //////////
    // Create the mutex.  The mutex is allocated in the unlocked state.
    // <i>name</i> should be a string which describes the primitive
    // for debugging purposes.  
    Mutex(const std::string& name);

    //////////
    // Destroy the mutex.  Destroying the mutex in the locked state has
    // undefined results.
    ~Mutex();

    //////////
    // Acquire the Mutex.  This method will block until the mutex is acquired.
    void lock();

    //////////
    // Attempt to acquire the Mutex without blocking.  Returns true if the
    // mutex was available and actually acquired.
    bool tryLock();

    //////////
    // Release the Mutex, unblocking any other threads that are blocked
    // waiting for the Mutex.
    void unlock();
};

#endif
