
#include "LockUnlock.h"
#include "CrowdMonitor.h"
#include "Mutex.h"

/*
 * Utility class to lock and unlock a <b>Mutex</b> or a <b>CrowdMonitor</b>.
 * The constructor of the <b>LockUnlock</b> object will lock the mutex
 * (or <b>CrowdMonitor</b>), and the destructor will unlock it.
 * <p>
 * This would be used like this:
 * <blockquote><pre>
 * {
 * <blockquote>LockUnlock mlock(&mutex);  // Acquire the mutex
 *     ... critical section ...</blockquote>
 * } // mutex is released when mlock goes out of scope
 * </pre></blockquote>
 */

LockUnlock::LockUnlock(Mutex* mutex): mutex(mutex), monitor(0) {
    mutex->lock();
}

LockUnlock::LockUnlock(CrowdMonitor* crowdMonitor, Which action) : monitor(crowdMonitor),
    action(action), mutex(0) {
    if(action==Read)
        monitor->read_lock();
    else
        monitor->write_lock();
}

LockUnlock::~LockUnlock() {
    if(mutex)
        mutex->unlock();
    else if(action==Read)
        monitor->read_unlock();
    else
        monitor->write_unlock();
}

