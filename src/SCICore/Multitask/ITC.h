
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

#include <SCICore/share/share.h>

namespace SCICore {
namespace Multitask {

struct Barrier_private;
struct ConditionVariable_private;
struct CrowdMonitor_private;
struct Mutex_private;
struct LibMutex_private;
struct Semaphore_private;


class SCICORESHARE Mutex {
    Mutex_private* priv;
    friend class ConditionVariable;
public:
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
    int try_lock();
};

class SCICORESHARE LibMutex {
    LibMutex_private* priv;
public:
    LibMutex();
    ~LibMutex();
    void lock();
    void unlock();
    int try_lock();
};

class SCICORESHARE Semaphore {
    Semaphore_private* priv;
public:
    Semaphore(int count);
    ~Semaphore();
    void up();
    void down();
    int try_down();
};

class SCICORESHARE Barrier {
    Barrier_private* priv;
public:
    Barrier();
    ~Barrier();
    void wait(int n);
};

class SCICORESHARE ConditionVariable {
    ConditionVariable_private* priv;
public:
    ConditionVariable();
    ~ConditionVariable();
    void wait(Mutex&);
    void cond_signal();
    void broadcast();
};

class SCICORESHARE CrowdMonitor {
    CrowdMonitor_private* priv;
public:
    CrowdMonitor();
    ~CrowdMonitor();
    void read_lock();
    void read_unlock();

    void write_lock();
    void write_unlock();
};

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:37  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:06  mcq
// Initial commit
//
// Revision 1.4  1999/06/30 21:49:04  moulding
// added SHARE to enable win32 shared libraries (dll's).
// .
//
// Revision 1.3  1999/05/06 19:56:21  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:27  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#include <SCICore/Multitask/Mailbox.h>

#endif /* SCI_Multitask_ITC_h */
