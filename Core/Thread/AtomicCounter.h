
// $Id$

/*
 *  AtomicCounter.h: Thread-safe integer variable
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_AtomicCounter_h
#define SCICore_Thread_AtomicCounter_h

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

   Not that this implementation does not offer an operator=, but
   instead uses a "set" method.  This is to avoid the inadvertant
   use of a statement like: x=x+2, which would NOT be thread safe.

PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/Mutex.h>

namespace SCICore {
    namespace Thread {
	class AtomicCounter {
	    const char* d_name;
	    Mutex d_lock;
	    int d_value;
	public:
	    //////////
	    // Create an atomic counter with an unspecified initial value.
	    // <tt>name</tt> should be a static string which describes the
	    // primitive for debugging purposes.
	    AtomicCounter(const char* name);

	    //////////
	    // Create an atomic counter with an initial value.  name should
	    // be a static string which describes the primitive for debugging
	    // purposes.
	    AtomicCounter(const char* name, int value);

	    //////////
	    // Destroy the atomic counter.
	    ~AtomicCounter();

	    //////////
	    // Allows the atomic counter to be used in expressions like
	    // a normal integer.  Note that multiple calls to this function
	    // may return different values if other threads are manipulating
	    // the counter.
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

	    //////////
	    // Set the counter to a new value
	    void set(int);
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/08/25 02:37:54  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
