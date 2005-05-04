/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  FutureValue: Delayed return values
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_FutureValue_h
#define Core_Thread_FutureValue_h

#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>

namespace SCIRun {
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
   
****************************************/
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

template<class Item>
FutureValue<Item>::FutureValue(const char* name)
    : name(name), sema("FutureValue semaphore", 0)
{
}

template<class Item>
FutureValue<Item>::~FutureValue()
{
}

template<class Item>
Item
FutureValue<Item>::receive()
{
  int s=Thread::couldBlock(name);
  sema.down();
  Thread::couldBlockDone(s);
  return value;
} // End namespace SCIRun

template<class Item>
void
FutureValue<Item>::send(const Item& reply)
{
  value=reply;
  sema.up();
}
} // End namespace SCIRun

#endif

