
/*
 *  Guard: Automatically lock/unlock a mutex or crowdmonitor.
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Guard_h
#define Core_Thread_Guard_h

#include <Core/share/share.h>

namespace SCIRun {

class Mutex;
class CrowdMonitor;

/**************************************

  CLASS
  Guard

  KEYWORDS
  Thread

  DESCRIPTION
  Utility class to lock and unlock a <b>Mutex</b> or a <b>CrowdMonitor</b>.
  The constructor of the <b>Guard</b> object will lock the mutex
  (or <b>CrowdMonitor</b>), and the destructor will unlock it.
  <p>
  This would be used like this:
  <blockquote><pre>
  {
  <blockquote>Guard mlock(&mutex);  // Acquire the mutex
  ... critical section ...</blockquote>
  } // mutex is released when mlock goes out of scope
  </pre></blockquote>
   
****************************************/
class SCICORESHARE Guard {
public:
  //////////
  // Attach the <b>Guard</b> object to the <i>mutex</i>, and
  // acquire the mutex.
  Guard(Mutex* mutex);
  enum Which {
    Read,
    Write
  };
    
  //////////
  // Attach the <b>Guard</b> to the <i>CrowdMonitor</pre> and
  // acquire one of the locks.  If <i>action</i> is
  // <b>Guard::Read</b>, the read lock will be acquired, and if
  // <i>action</i> is <b>Write</b>, then the write lock will be
  // acquired.  The appropriate lock will then be released by the
  // destructor
  Guard(CrowdMonitor* crowdMonitor, Which action);
    
  //////////
  // Release the lock acquired by the constructor.
  ~Guard();
private:
  Mutex* mutex_;
  CrowdMonitor* monitor_;
  Which action_;

  // Cannot copy them
  Guard(const Guard&);
  Guard& operator=(const Guard&);
};
} // End namespace SCIRun

#endif

