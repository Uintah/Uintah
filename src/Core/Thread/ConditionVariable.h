
/*
 *  ConditionVariable: Condition variable primitive
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

#ifndef SCICore_Thread_ConditionVariable_h
#define SCICore_Thread_ConditionVariable_h

namespace SCICore {
    namespace Thread {
	class ConditionVariable_private;
	class Mutex;
/**************************************
 
CLASS
   ConditionVariable
   
KEYWORDS
   Thread
   
DESCRIPTION
   Condition variable primitive.  When a thread calls the
   <i>wait</i> method,which will block until another thread calls
   the <i>conditionSignal</i> or <i>conditionBroadcast</i> methods.  When
   there are multiple threads waiting, <i>conditionBroadcast</i> will unblock
   all of them, while <i>conditionSignal</i> will unblock only one (an
   arbitrary one) of them.  This primitive is used to allow a thread
   to wait for some condition to exist, such as an available resource.
   The thread waits for that condition, until it is unblocked by another
   thread that caused the condition to exist (<i>i.e.</i> freed the
   resource).
   
****************************************/

	class ConditionVariable {
	public:
	    //////////
	    // Create a condition variable. <i>name</i> should be a static
	    // string which describes the primitive for debugging purposes.
	    ConditionVariable(const char* name);
    
	    //////////
	    // Destroy the condition variable
	    ~ConditionVariable();
    
	    //////////
	    // Wait for a condition.  This method atomically unlocks
	    // <b>mutex</b>, and blocks.  The <b>mutex</b> is typically
	    // used to guard access to the resource that the thread is
	    // waiting for.
	    void wait(Mutex& m);
    
	    //////////
	    // Signal a condition.  This will unblock one of the waiting
	    // threads. No guarantee is made as to which thread will be
	    // unblocked, but thread implementations typically give
	    // preference to the thread that has waited the longest.
	    void conditionSignal();

	    //////////
	    // Signal a condition.  This will unblock all of the waiting
	    // threads. Note that only the number of waiting threads will
	    // be unblocked. No guarantee is made that these are the same
	    // N threads that were blocked at the time of the broadcast.
	    void conditionBroadcast();

	private:
	    const char* d_name;
	    ConditionVariable_private* d_priv;

	    // Cannot copy them
	    ConditionVariable(const ConditionVariable&);
	    ConditionVariable& operator=(const ConditionVariable&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.7  1999/09/02 16:52:41  sparker
// Updates to cocoon documentation
//
// Revision 1.6  1999/08/28 03:46:47  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 19:00:47  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:37:55  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
