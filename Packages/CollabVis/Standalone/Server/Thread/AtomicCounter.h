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
 *  AtomicCounter: Thread-safe integer variable
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_AtomicCounter_h
#define Core_Thread_AtomicCounter_h

#include <Core/share/share.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

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
class SCICORESHARE AtomicCounter {
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
  const char* name_;
  AtomicCounter_private* priv_;

  // Cannot copy them
  AtomicCounter(const AtomicCounter&);
  AtomicCounter& operator=(const AtomicCounter&);
};
} // End namespace SCIRun

#endif

