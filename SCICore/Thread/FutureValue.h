
// $Id$

/*
 *  FutureValue.h: Delayed return values
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_FutureValue_h
#define SCICore_Thread_FutureValue_h

/**************************************
 
CLASS
   FutureValue
   
KEYWORDS
   FutureValue
   
DESCRIPTION
   Creates a single slot for some return value.  The <i>wait</i> method
   waits for a value to be sent from another thread via the <i>reply</i>
   method.  This is typically used to provide a simple means of returning
   data from a server thread.  An <b>FutureValue</b> object is created on the
   stack, and some request is sent (usually via a <b>Mailbox</b>) to a server
   thread.  Then the thread will block in <i>wait</i> until the server thread
   receives the message and responds using <i>reply</i>.
  
   <p><b>FutureValue</b> is a one-shot wait/reply pair - a new
   <b>FutureValue</b> object must be created for each reply, and these are
   typically created on the stack.  Only a single thread should
   call <i>wait</i> and a single thread shuold call <i>reply</i>.
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Semaphore.h>

namespace SCICore {
    namespace Thread {
	template<class Item> class FutureValue {
	    const char* name;
	    Item value;
	    Semaphore sema;
	public:
	    //////////
	    // Create the FutureValue object.  <i>name</i> should be a
	    // static string which describes the primitive for debugging
	    // purposes.
	    FutureValue(const char* name);

	    //////////
	    // Destroy the object
	    ~FutureValue();

	    //////////
	    // Wait until the reply is sent by anothe thread, then return
	    // that reply.
	    Item wait();

	    //////////
	    // Send the reply to the waiting thread.
	    void reply(const Item& reply);
	};
    }
}
template<class Item>
SCICore::Thread::FutureValue<Item>::FutureValue(const char* name)
    : name(name), sema("FutureValue semaphore", 0)
{
}

template<class Item>
SCICore::Thread::FutureValue<Item>::~FutureValue()
{
}

template<class Item>
Item
SCICore::Thread::FutureValue<Item>::wait()
{
  int s=Thread::couldBlock(name);
  sema.down();
  Thread::couldBlock(s);
  return value;
}

template<class Item>
void
SCICore::Thread::FutureValue<Item>::reply(const Item& reply)
{
  value=reply;
  sema.up();
}

#endif

//
// $Log$
// Revision 1.1  1999/08/25 02:37:55  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
