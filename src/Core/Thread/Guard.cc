
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
