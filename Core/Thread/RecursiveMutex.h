
#ifndef SCI_THREAD_RECURSIVEMUTEX_H
#define SCI_THREAD_RECURSIVEMUTEX_H 1

/**************************************
 
CLASS
   RecursiveMutex
   
KEYWORDS
   RecursiveMutex
   
DESCRIPTION
   Provides a recursive <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
   <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
   Nested calls to <b>lock()</b> by the same thread are acceptable,
   but must be matched with calls to <b>unlock()</b>.  This class
   may be less efficient that the <b>Mutex</b> class, and should not
   be used unless the recursive lock feature is really required.
 
PATTERNS


WARNING
   
****************************************/

class Thread;
class RecursiveMutex_private;
#include "Mutex.h"
#include <string>

class RecursiveMutex {
    Mutex d_my_lock;
    RecursiveMutex_private* d_priv;
    Thread* d_owner;
    int d_lock_count;
public:
    //////////
    // Create the Mutex.  The Mutex is allocated in the unlocked state.
    // <i>name</i> should be a string which describe the primitive
    // for debugging purposes.
    RecursiveMutex(const std::string& name);

    //////////
    // Destroy the Mutex.  Destroying a Mutex in the locked state has
    // undefined results.
    ~RecursiveMutex();

    //////////
    // Acquire the Mutex.  This method will block until the Mutex is acquired.
    void lock();

    //////////
    // Release the Mutex, unblocking any other threads that are blocked
    // waiting for the Mutex.
    void unlock();
};

#endif






