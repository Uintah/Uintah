
/*
 *  Runnable: The base class for all threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Runnable_h
#define Core_Thread_Runnable_h

#include <Core/share/share.h>

namespace SCIRun {

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
	class SCICORESHARE Runnable {
	protected:
	    friend class Thread;
	    Thread* my_thread_;
    
	    //////////
	    // Create a new runnable, and initialize it's state.
	    Runnable();

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
	    // Cannot copy them
	    Runnable(const Runnable&);
	    Runnable& operator=(const Runnable&);
	};
} // End namespace SCIRun

#endif





