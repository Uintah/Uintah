
/*
 *  FutureValue: Delayed return values
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

#ifndef SCICore_Thread_FutureValue_h
#define SCICore_Thread_FutureValue_h

/**************************************
 
CLASS
   FutureValue
   
KEYWORDS
   Thread
   
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

#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>

namespace SCICore {
    namespace Thread {
	template<class Item> class FutureValue {
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
	    Item receive();

	    //////////
	    // Send the reply to the waiting thread.
	    void send(const Item& reply);

	private:
	    const char* name;
	    Item value;
	    Semaphore sema;

	    // Cannot copy them
	    FutureValue(const FutureValue<Item>&);
	    FutureValue<Item>& operator=(const FutureValue<Item>&);
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
SCICore::Thread::FutureValue<Item>::receive()
{
  int s=Thread::couldBlock(name);
  sema.down();
  Thread::couldBlockDone(s);
  return value;
}

template<class Item>
void
SCICore::Thread::FutureValue<Item>::send(const Item& reply)
{
  value=reply;
  sema.up();
}

#endif

//
// $Log$
// Revision 1.4  1999/08/28 17:54:53  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/28 03:46:47  sparker
// Final updates before integration with PSE
//
// Revision 1.2  1999/08/25 19:00:48  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.1  1999/08/25 02:37:55  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
