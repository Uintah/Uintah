
/*
 *  AtomicCounter: Thread-safe integer variable
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

#ifndef SCICore_Thread_AtomicCounter_h
#define SCICore_Thread_AtomicCounter_h

namespace SCICore {
    namespace Thread {
	class AtomicCounter_private;

/**************************************
 
CLASS
   AtomicCounter
   
KEYWORDS
   Thread
   
DESCRIPTION
   Provides a simple atomic counter.  This will work just like an
   integer, but guarantees atomicty of the ++ and -- operators.
   Despite their convenience, you do not want to make a large number
   of these objects.  See also WorkQueue.

   Not that this implementation does not offer an operator=, but
   instead uses a "set" method.  This is to avoid the inadvertant
   use of a statement like: x=x+2, which would NOT be thread safe.

****************************************/
	class AtomicCounter {
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
	    // This does not return AtomicCounter& like a normal ++
	    // operator would, because it would destroy atomicity
	    int operator++();
    
	    //////////
	    //	Increment the counter and return the old value
	    int operator++(int);

	    //////////
	    // Decrement the counter and return the new value
	    // This does not return AtomicCounter& like a normal --
	    // operator would, because it would destroy atomicity
	    int operator--();
    
	    //////////
	    // Decrement the counter and return the old value
	    int operator--(int);

	    //////////
	    // Set the counter to a new value
	    void set(int);

	private:
	    const char* d_name;
	    AtomicCounter_private* d_priv;

	    // Cannot copy them
	    AtomicCounter(const AtomicCounter&);
	    AtomicCounter& operator=(const AtomicCounter&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.8  1999/09/02 16:52:41  sparker
// Updates to cocoon documentation
//
// Revision 1.7  1999/08/29 00:46:59  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/28 03:46:46  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:46  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:54  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
