
#include "CrowdMonitor.h"

/*
 * Multiple reader, single writer synchronization primitive.  Some
 * applications do not need the strict level of synchronization
 * provided by the <b>Mutex</b>.  The <b>CrowdMonitor</b> relaxes
 * the synchronization by allowing multiple threads access to a
 * resource (usually a data area), on the condition that the thread
 * will only read the data.  When a thread needs to write the data,
 * it can access the monitor in write mode (using <i>writeLock</i>).
 * At any given time, one writer thread can be active, or multiple
 * reader threads can be active.  <b>CrowdMonitor</b> guards against
 * multiple writers accessing a data, and against a thread writing
 * to the data while other threads are reading it.

 * <p> Calling <i>read_lock</i> within a <i>write_lock/writeUnlock</i>
 * section may result in a deadlock.  Likewise, calling <i>writeLock</i>
 * within a <i>readLock/readUnlock</i> section may result in a deadlock.
 * Calling <i>readUnlock</i> or <i>writeUnlock</i> when the lock is
 * not held is not legal and may result in undefined behavior.
 */

CrowdMonitor::CrowdMonitor(const std::string& name)
    : d_name(name), d_writeWaiters("CrowdMonitor write condition"),
      d_readWaiters("CrowdMonitor read condition"),
      d_lock("CrowdMonitor lock")
{
    d_numReadersWaiting=0;
    d_numWritersWaiting=0;
    d_numReaders=0;
    d_numWriters=0;
}

CrowdMonitor::~CrowdMonitor()
{
}

void CrowdMonitor::readLock()
{
    d_lock.lock();
    while(d_numWriters > 0){
        d_numReadersWaiting++;
        d_readWaiters.wait(d_lock);
        d_numReadersWaiting--;
    }
    d_numReaders++;
    d_lock.unlock();
}

void CrowdMonitor::readUnlock()
{
    d_lock.lock();
    d_numReaders--;
    if(d_numReaders == 0 && d_numWritersWaiting > 0)
        d_writeWaiters.conditionSignal();
    d_lock.unlock();
}

void CrowdMonitor::writeLock()
{
    d_lock.lock();
    while(d_numWriters || d_numReaders){
        // Have to wait...
        d_numWritersWaiting++;
        d_writeWaiters.wait(d_lock);
        d_numWritersWaiting--;
    }
    d_numWriters++;
    d_lock.unlock();
}

void CrowdMonitor::writeUnlock()
{
    d_lock.lock();
    d_numWriters--;
    if(d_numWritersWaiting)
        d_writeWaiters.conditionSignal(); // Wake one of them up...
    else if(d_numReadersWaiting)
        d_readWaiters.conditionBroadcast(); // Wake all of them up...
    d_lock.unlock();
}

