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
  mutable Semaphore* wait_; // This may be modified by Thread::parallel
  friend class Thread;

private:
  // Cannot copy them
  ParallelBase(const ParallelBase&);
  ParallelBase& operator=(const ParallelBase&);
};
} // End namespace SCIRun

#endif



