 
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

#include <Multitask/Task.h>
#include <Multitask/ITC.h>

class TCLTask : public Task {
    int argc;
    char** argv;
    Semaphore cont;
    Semaphore start;
protected:
    virtual int body(int);
    friend void wait_func(void*);
    void mainloop_wait();
public:
    TCLTask(int argc, char* argv[]);
    virtual ~TCLTask();
    static Task* get_owner();
    static void lock();
    static int try_lock();
    static void unlock();
    void mainloop_waitstart();
    void release_mainloop();
};

#endif
