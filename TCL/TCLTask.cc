 
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

#include <Multitask/ITC.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCL.h>

#include <iostream.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

extern "C" int tkMain(int argc, char** argv, void (*)(void*), void*);
extern Tcl_Interp* the_interp;
typedef void (Tcl_LockProc)();
extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);

static Mutex* tlock=0;
static Task* owner;
static int lock_count;

static void do_lock()
{
    if(owner == Task::self()){
	lock_count++;
	//cerr << "Recursively locked...\n";
	return;
    }
    tlock->lock();
    lock_count=1;
    owner=Task::self();
    //cerr << "Locked: owner=" << owner << ", count=" << lock_count << endl;
}

static void do_unlock()
{
    ASSERT(lock_count>0);
    ASSERT(Task::self() == owner);
    if(--lock_count == 0){
	owner=0;
	//cerr << "Unlocked, count=" << lock_count << ", owner=" << owner << ", self=" << Task::self() << endl;
	tlock->unlock();
    } else {
	//cerr << "Recursively unlocked" << endl;
    }
}

static int x_error_handler(Display* dpy, XErrorEvent* error)
{
    char msg[200];
    XGetErrorText(dpy, error->error_code, msg, 200);
    cerr << "X Error: " << msg << endl;
    abort();
    return 0; // Never reached...
}

static int exitproc(ClientData, Tcl_Interp*, int, char* [])
{
    Task::exit_all(0);
    return TCL_OK; // not reached
}

TCLTask::TCLTask(int argc, char* argv[])
: Task("TCLTask", 1), argc(argc), argv(argv), start(0), cont(0)
{
    // Setup the error handler to catch errors...
    // The default one exits, and makes it very hard to 
    // track down errors.  We need core dumps!
    XSetErrorHandler(x_error_handler);

    if(!tlock)
	tlock=scinew Mutex;
    Tcl_SetLock(do_lock, do_unlock);
}

TCLTask::~TCLTask()
{
}

void wait_func(void* thatp)
{
    TCLTask* that=(TCLTask*)thatp;
    that->mainloop_wait();
}

int TCLTask::body(int)
{
    tkMain(argc, argv, wait_func, (void*)this);
    return 0;
}

void TCLTask::mainloop_waitstart()
{
    start.down();
}

void TCLTask::release_mainloop()
{
    cont.up();
}

void TCLTask::mainloop_wait()
{
    TCL::initialize();
    Tcl_CreateCommand(the_interp, "exit", exitproc, 0, 0);

    // The main program will want to know that we are started...
    start.up();

    // Wait for the main program to tell us that all initialization
    // has occurred...
    cont.down();
}

void TCLTask::lock()
{
    do_lock();
}

void TCLTask::unlock()
{
    do_unlock();
}

int TCLTask::try_lock()
{
    if(owner == Task::self()){
	lock_count++;
	return 1;
    }
    if(tlock->try_lock()){
	lock_count=1;
	owner=Task::self();
//	cerr << "Locked (try): owner=" << owner << ", count=" << lock_count << endl;
	return 1;
    } else {
//	cerr << "Try Lock failed" << endl;
	return 0;
    }
}
