
/*
 *  CrowdMonitor: Multiple reader/single writer locks
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

#ifndef SCICore_Thread_CrowdMonitor_h
#define SCICore_Thread_CrowdMonitor_h

#include <SCICore/share/share.h>

#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/ConditionVariable.h>

namespace SCICore {
    namespace Thread {
	class CrowdMonitor_private;
/**************************************
 
CLASS
   CrowdMonitor
   
KEYWORDS
   Thread
   
DESCRIPTION
   Multiple reader, single writer synchronization primitive.  Some
   applications do not need the strict level of synchronization
   provided by the <b>Mutex</b>.  The <b>CrowdMonitor</b> relaxes
   the synchronization by allowing multiple threads access to a
   resource (usually a data area), on the condition that the thread
   will only read the data.  When a thread needs to write the data,
   it can access the monitor in write mode (using <i>writeLock</i>).
   At any given time, one writer thread can be active, or multiple
   reader threads can be active.  <b>CrowdMonitor</b> guards against
   multiple writers accessing a data, and against a thread writing
   to the data while other threads are reading it.

WARNING
   <p> Calling <i>readLock</i> within a <i>writeLock/write_unlock</i>
   section may result in a deadlock.  Likewise, calling <i>writeLock</i>
   within a <i>readLock/readUnlock</i> section may result in a deadlock.
   Calling <i>readUnlock</i> or <i>writeUnlock</i> when the lock is
   not held is not legal and may result in undefined behavior.
   
****************************************/

	class SCICORESHARE CrowdMonitor {
	public:
	    //////////
	    // Create and initialize the CrowdMonitor. <i>name</i> should
	    // be a static which describes the primitive for debugging
	    // purposes.
	    CrowdMonitor(const char* name);
    
	    //////////
	    // Destroy the CrowdMonitor.
	    ~CrowdMonitor();
    
	    //////////
	    // Acquire the read-only lock associated with this
	    // <b>CrowdMonitor</b>. Multiple threads may hold the
	    // read-only lock simultaneously.
	    void readLock();
    
	    //////////
	    // Release the read-only lock obtained from <i>readLock</i>.
	    // Undefined behavior may result when <i>readUnlock</i> is
	    // called and a <i>readLock</i> is not held by the calling
	    // Thread.
	    void readUnlock();
    
	    //////////
	    // Acquire the write lock associated with this
	    // <b>CrowdMonitor</b>. Only one thread may hold the write
	    // lock, and during the time that this lock is not held, no
	    // threads may hold the read-only lock.
	    void writeLock();

	    //////////
	    // Release the write-only lock obtained from <i>writeLock</i>.
	    // Undefined behavior may result when <i>writeUnlock</i> is
	    // called and a <i>writeLock</i> is not held by the calling
	    // Thread.
	    void writeUnlock();

	private:
	    const char* d_name;
	    CrowdMonitor_private* d_priv;

	    // Cannot copy them
	    CrowdMonitor(const CrowdMonitor&);
	    CrowdMonitor& operator=(const CrowdMonitor&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.9  1999/11/02 06:11:02  moulding
// added some #includes to help the visual c++ compiler
//
// Revision 1.8  1999/09/24 18:55:06  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.7  1999/09/02 16:52:42  sparker
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
