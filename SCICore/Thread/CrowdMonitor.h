
#ifndef SCI_THREAD_CROWDMONITOR_H
#define SCI_THREAD_CROWDMONITOR_H 1

/**************************************
 
CLASS
   CrowdMonitor
   
KEYWORDS
   CrowdMonitor
   
DESCRIPTION
   Multiple reader, single writer synchronization primitive.  Some
   applications do not need the strict level of synchronization
   provided by the <b>Mutex</b>.  The <b>CrowdMonitor</b> relaxes
   the synchronization by allowing multiple threads access to a
   resource (usually a data area), on the condition that the thread
   will only read the data.  When a thread needs to write the data,
   it can access the monitor in write mode (using <i>writeLock</i>).
   At any given time, one writer thread can be active, or multiple
   reader threads can be active.  <b>CrowdMonitor</b> guards against
   multiple writers accessing a data, and against a thread writing
   to the data while other threads are reading it.

PATTERNS


WARNING
   <p> Calling <i>readLock</i> within a <i>writeLock/write_unlock</i>
   section may result in a deadlock.  Likewise, calling <i>writeLock</i>
   within a <i>readLock/readUnlock</i> section may result in a deadlock.
   Calling <i>readUnlock</i> or <i>writeUnlock</i> when the lock is
   not held is not legal and may result in undefined behavior.
   
****************************************/

#include "Mutex.h"
#include "ConditionVariable.h"
#include <string>

class CrowdMonitor_private;

class CrowdMonitor {
    std::string d_name;
    ConditionVariable d_write_waiters;
    ConditionVariable d_read_waiters;
    int d_num_readers_waiting;
    int d_num_writers_waiting;
    int d_num_readers;
    int d_num_writers;
    Mutex d_lock;
public:
    //////////
    // Create and initialize the CrowdMonitor. <i>name</i> should be a
    // static which describes the primitive for debugging purposes.
    CrowdMonitor(const std::string& name);
    
    //////////
    // Destroy the CrowdMonitor.
    ~CrowdMonitor();
    
    //////////
    // Acquire the read-only lock associated with this <b>CrowdMonitor</b>.
    // Multiple threads may hold the read-only lock.
    void readLock();
    
    //////////
    // Release the read-only lock obtained from <i>readLock</i>.  Undefined
    // behavior may result when <i>readUnlock</i> is called and a
    // <i>readLock</i> is not held.  
    void readUnlock();
    
    //////////
    // Acquire the write lock associated with this <b>CrowdMonitor</b>.
    // Only one thread may hold the write lock, and during the time that
    // this lock is not held, no threads may hold the read-only lock.
    void writeLock();

    //////////
    // Release the write-only lock obtained from <i>writeLock</i>.
    // Undefined behavior may result when <i>writeUnlock</i> is called
    // and a <i>writeLock</i> is not held.
    void writeUnlock();
};

#endif


