
#include "Guard.h"
#include "CrowdMonitor.h"
#include "Mutex.h"

/*
 * Utility class to lock and unlock a <b>Mutex</b> or a <b>CrowdMonitor</b>.
 * The constructor of the <b>Guard</b> object will lock the mutex
 * (or <b>CrowdMonitor</b>), and the destructor will unlock it.
 * <p>
 * This would be used like this:
 * <blockquote><pre>
 * {
 * <blockquote>Guard mlock(&mutex);  // Acquire the mutex
 *     ... critical section ...</blockquote>
 * } // mutex is released when mlock goes out of scope
 * </pre></blockquote>
 */

Guard::Guard(Mutex* mutex)
    : d_mutex(mutex), d_monitor(0)
{
    d_mutex->lock();
}

Guard::Guard(CrowdMonitor* crowdMonitor, Which action) 
    : d_mutex(0), d_monitor(crowdMonitor), action(action)
{
    if(action==Read)
        d_monitor->readLock();
    else
        d_monitor->writeLock();
}

Guard::~Guard()
{
    if(d_mutex)
        d_mutex->unlock();
    else if(action==Read)
        d_monitor->readUnlock();
    else
        d_monitor->writeUnlock();
}

