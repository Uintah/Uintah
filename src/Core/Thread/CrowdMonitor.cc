
#include "CrowdMonitor.h"

/*
 * Multiple reader, single writer synchronization primitive.  Some
 * applications do not need the strict level of synchronization
 * provided by the <b>Mutex</b>.  The <b>CrowdMonitor</b> relaxes
 * the synchronization by allowing multiple threads access to a
 * resource (usually a data area), on the condition that the thread
 * will only read the data.  When a thread needs to write the data,
 * it can access the monitor in write mode (using <i>write_lock</i>).
 * At any given time, one writer thread can be active, or multiple
 * reader threads can be active.  <b>CrowdMonitor</b> guards against
 * multiple writers accessing a data, and against a thread writing
 * to the data while other threads are reading it.

 * <p> Calling <i>read_lock</i> within a <i>write_lock/write_unlock</i>
 * section may result in a deadlock.  Likewise, calling <i>write_lock</i>
 * within a <i>read_lock/read_unlock</i> section may result in a deadlock.
 * Calling <i>read_unlock</i> or <i>write_unlock</i> when the lock is
 * not held is not legal and may result in undefined behavior.
 */

CrowdMonitor::CrowdMonitor(const char* name)
     : name(name), write_waiters("CrowdMonitor write condition"),
   read_waiters("CrowdMonitor read condition"),
   lock("CrowdMonitor lock") {
    nreaders_waiting=0;
    nwriters_waiting=0;
    nreaders=0;
    nwriters=0;
}

CrowdMonitor::~CrowdMonitor() {}

void CrowdMonitor::read_lock() {
    lock.lock();
    while(nwriters > 0){
        nreaders_waiting++;
        read_waiters.wait(lock);
        nreaders_waiting--;
    }
    nreaders++;
    lock.unlock();
}

void CrowdMonitor::read_unlock() {
    lock.lock();
    nreaders--;
    if(nreaders == 0 && nwriters_waiting > 0)
        write_waiters.cond_signal();
    lock.unlock();
}

void CrowdMonitor::write_lock() {
    lock.lock();
    while(nwriters || nreaders){
        // Have to wait...
        nwriters_waiting++;
        write_waiters.wait(lock);
        nwriters_waiting--;
    }
    nwriters++;
    lock.unlock();
}

void CrowdMonitor::write_unlock() {
    lock.lock();
    nwriters--;
    if(nwriters_waiting)
        write_waiters.cond_signal(); // Wake one of them up...
    else if(nreaders_waiting)
        read_waiters.cond_broadcast(); // Wake all of them up...
    lock.unlock();
}

