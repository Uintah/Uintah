
/*
 *  ParallelBase: Helper class to instantiate several threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_ParallelBase_h
#define Core_Thread_ParallelBase_h

#include <Core/share/share.h>

namespace SCIRun {

class Semaphore;
/**************************************

 CLASS
 ParallelBase

 KEYWORDS
 Thread

 DESCRIPTION
 Helper class for Parallel class.  This will never be used
 by a user program.  See <b>Parallel</b> instead.
   
****************************************/
class SCICORESHARE ParallelBase {
public:
  //////////
  // <i>The thread body</i>
  virtual void run(int proc)=0;

protected:
  ParallelBase();
  virtual ~ParallelBase();
  mutable Semaphore* d_wait; // This may be modified by Thread::parallel
  friend class Thread;

private:
  // Cannot copy them
  ParallelBase(const ParallelBase&);
  ParallelBase& operator=(const ParallelBase&);
};
} // End namespace SCIRun

#endif



