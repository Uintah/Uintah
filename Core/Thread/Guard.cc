
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
    : d_mutex(mutex), d_monitor(0)
{
    d_mutex->lock();
}

Guard::Guard(CrowdMonitor* crowd_monitor, Which action) 
    : d_mutex(0), d_monitor(crowd_monitor), d_action(action)
{
    if(d_action==Read)
        d_monitor->readLock();
    else
        d_monitor->writeLock();
}

Guard::~Guard()
{
    if(d_mutex)
        d_mutex->unlock();
    else if(d_action==Read)
        d_monitor->readUnlock();
    else
        d_monitor->writeUnlock();
}


} // End namespace SCIRun
