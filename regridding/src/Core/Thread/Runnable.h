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
 *  Runnable: The base class for all threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 */

#ifndef Core_Thread_Runnable_h
#define Core_Thread_Runnable_h

namespace SCIRun {

  class Thread;

/**************************************
 
CLASS
   Runnable
   
KEYWORDS
   Thread
   
DESCRIPTION
   
   This class should be a base class for any class which is to be
   attached to a thread.  It provides a <i>run</i> pure virtual method
   which should be overridden to provide the thread body.  When this
   method returns, or the thread calls <i>Thread::exit</i>, the
   thread terminates.  A <b>Runnable</b> should be attached to
   only one thread.
  
   <p> It is very important that the <b>Runnable</b> object (or any
   object derived from it) is never explicitly deleted.  It will be
   deleted by the <b>Thread</b> to which it is attached, when the
   thread terminates.  The destructor will be executed in the context
   of this same thread.
   
****************************************/
class Runnable {

  protected:

    friend class Thread;
    Thread* my_thread_;

    //////////
    // Create a new runnable, and initialize it's state.
    Runnable(bool delete_on_exit = true);

    //////////
    // The runnable destructor.  See the note above about deleting any
    // object derived from runnable.
    virtual ~Runnable();

    //////////
    // This method will be overridden to implement the main body
    // of the thread.  This method will called when the runnable
    // is attached to a <b>Thread</b> object, and will be executed
    // in a new context.
    virtual void run()=0;

  private:

    bool delete_on_exit;

    // Cannot copy them
    Runnable(const Runnable&);
    Runnable& operator=(const Runnable&);
};
}
  // End namespace SCIRun

#endif





