 
/*
 *  TCLTask.h:  Handle TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TCLTask_h
#define SCI_project_TCLTask_h 1

#include <Core/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>

namespace SCIRun {

class SCICORESHARE TCLTask : public Runnable {
    int argc;
    char** argv;
    Semaphore cont;
    Semaphore start;
protected:
    virtual void run();
    friend void wait_func(void*);
    void mainloop_wait();
public:
    TCLTask(int argc, char* argv[]);
    virtual ~TCLTask();
    static Thread* get_owner();
    static void lock();
    static int try_lock();
    static void unlock();
    void mainloop_waitstart();
    void release_mainloop();
};

} // End namespace SCIRun


#endif
