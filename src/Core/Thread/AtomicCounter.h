/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#ifndef Core_Thread_AtomicCounter_h
#define Core_Thread_AtomicCounter_h

namespace SCIRun {

struct AtomicCounter_private;

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
  AtomicCounter(const char* name, long long value);

  //////////
  // Destroy the atomic counter.
  ~AtomicCounter();

  //////////
  // Allows the atomic counter to be used in expressions like
  // a normal integer.  Note that multiple calls to this function
  // may return different values if other threads are manipulating
  // the counter.
  operator long long() const;

  //////////
  // Increment the counter and return the new value.
  // This does not return AtomicCounter& like a normal ++
  // operator would, because it would destroy atomicity
  long long operator++();

  //////////
  //	Increment the counter and return the old value
  long long operator++(int);

  //////////
  // Decrement the counter and return the new value
  // This does not return AtomicCounter& like a normal --
  // operator would, because it would destroy atomicity
  long long operator--();

  //////////
  // Decrement the counter and return the old value
  long long operator--(int);

  //////////
  // Set the counter to a new value
  void set(long long);

  private:
  const char* name_;
  AtomicCounter_private* priv_;

  long long value_;
  // Cannot copy them
  AtomicCounter(const AtomicCounter&);
  AtomicCounter& operator=(const AtomicCounter&);
};

}
  // End namespace SCIRun

#endif

