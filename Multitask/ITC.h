
/*
 *  ITC.h: Inter-task communication
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Multitask_ITC_h
#define SCI_Multitask_ITC_h 1

#ifdef __GNUG__
#pragma interface
#endif

struct Barrier_private;
struct ConditionVariable_private;
struct CrowdMonitor_private;
struct Mutex_private;
struct LibMutex_private;
struct Semaphore_private;


class Mutex {
    Mutex_private* priv;
    friend class ConditionVariable;
public:
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
    int try_lock();
};

class LibMutex {
    LibMutex_private* priv;
public:
    LibMutex();
    ~LibMutex();
    void lock();
    void unlock();
    int try_lock();
};

class Semaphore {
    Semaphore_private* priv;
public:
    Semaphore(int count);
    ~Semaphore();
    void up();
    void down();
    int try_down();
};

class Barrier {
    Barrier_private* priv;
public:
    Barrier();
    ~Barrier();
    void wait(int n);
};

class ConditionVariable {
    ConditionVariable_private* priv;
public:
    ConditionVariable();
    ~ConditionVariable();
    void wait(Mutex&);
    void cond_signal();
    void broadcast();
};

class CrowdMonitor {
    CrowdMonitor_private* priv;
public:
    CrowdMonitor();
    ~CrowdMonitor();
    void read_lock();
    void read_unlock();

    void write_lock();
    void write_unlock();
};

#include <Multitask/Mailbox.h>

#endif /* SCI_Multitask_ITC_h */
