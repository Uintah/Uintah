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

#include <Core/Thread/Guard.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Mutex.h>
namespace SCIRun {

//using Guard;

Guard::Guard(Mutex* mutex)
    : mutex_(mutex), monitor_(0)
{
  mutex_->lock();
}

Guard::Guard(CrowdMonitor* crowd_monitor, Which action) 
    : mutex_(0), monitor_(crowd_monitor), action_(action)
{
    if(action_ == Read)
        monitor_->readLock();
    else
        monitor_->writeLock();
}

Guard::~Guard()
{
    if (mutex_)
        mutex_->unlock();
    else if(action_==Read)
        monitor_->readUnlock();
    else
        monitor_->writeUnlock();
}


} // End namespace SCIRun
