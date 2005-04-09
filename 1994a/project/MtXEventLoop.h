
/*
 *  MtXEventLoop.h: The Event loop thread
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MtXEventLoop_h
#define SCI_project_MtXEventLoop_h 1

#include <Multitask/Task.h>
#include <X11/Intrinsic.h>
#include <X11/Xlib.h>

class Mutex;
class Semaphore;

class MtXEventLoop : public Task {
    XtAppContext context;
    Display* display;
    Screen* screen;
    int started;
    int pipe_fd[2];
    Mutex* mutex;
    Semaphore* sema;
    int lock_count;
    Task* token_owner;
    Task* eventloop_taskid;
protected:
    virtual int body(int);
    void process_token(int fd);
    friend void do_process_token(XtPointer,int*, XtInputId*);
public:
    MtXEventLoop();
    virtual ~MtXEventLoop();
    Display* get_display();
    Screen* get_screen();
    XtAppContext get_app();
    void wait_start();
    void lock();
    void unlock();
};

#endif
