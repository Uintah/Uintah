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
 *  CrowdMonitor: Multiple reader/single writer locks
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_CrowdMonitor_h
#define Core_Thread_CrowdMonitor_h

#include <Core/share/share.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>

namespace SCIRun {
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
	    const char* name_;
	    CrowdMonitor_private* priv_;

	    // Cannot copy them
	    CrowdMonitor(const CrowdMonitor&);
	    CrowdMonitor& operator=(const CrowdMonitor&);
	};
} // End namespace SCIRun

#endif

