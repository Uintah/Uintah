
#ifndef SCI_THREAD_ATOMICCOUNTER_H
#define SCI_THREAD_ATOMICCOUNTER_H 1

/**************************************
 
CLASS
   AtomicCounter
   
KEYWORDS
   AtomicCounter
   
DESCRIPTION
   Provides a simple atomic counter.  This will work just like an
   integer, but guarantees atomicty of the ++ and -- operators.
   Despite their convenience, you do not want to make a large number
   of these objects.  See also WorkQueue.
PATTERNS


WARNING
   
****************************************/

#include "Mutex.h"
#include <string>

class AtomicCounter {
    std::string d_name;
    Mutex d_lock;
    int d_value;
public:
    //////////
    // Create an atomic counter with an unspecified initial value.
    // <tt>name</tt> should be a string which describes the
    // primitive for debugging purposes.
    AtomicCounter(const std::string& name);

    //////////
    // Create an atomic counter with an initial value.  name should be a 
    // string which describes the primitive for debugging purposes.
    AtomicCounter(const std::string& name, int value);

    //////////
    // Destroy the atomic counter.
    ~AtomicCounter();

    //////////
    // Allows the atomic counter to be used in expressions like a normal
    // integer.  Note that multiple calls to this function may return
    // different values if other threads are manipulating the counter.
    operator int() const;

    //////////
    // Increment the counter and return the new value.
    AtomicCounter& operator++();
    
    //////////
    //	Increment the counter and return the old value
    int operator++(int);

    //////////
    // Decrement the counter and return the new value
    AtomicCounter& operator--();
    
    //////////
    // Decrement the counter and return the old value
    int operator--(int);

    // Set the counter to a new value
    void set(int);
};

#endif

