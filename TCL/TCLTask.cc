 
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
#include <TCL/TCLTask.h>
#include <TCL/TCL.h>

#include <iostream.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

extern "C" int tkMain(int argc, char** argv);
extern "C" void Tk_FirstPrompt();
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
	return;
    }
    tlock->lock();
    lock_count=1;
    owner=Task::self();
//    cerr << "Locked: owner=" << owner << ", count=" << lock_count << endl;
}

static void do_unlock()
{
    ASSERT(lock_count>0);
    if(--lock_count == 0){
	owner=0;
//	cerr << "Unlocked" << endl;
	tlock->unlock();
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

TCLTask::TCLTask(int argc, char* argv[])
: Task("TCLTask", 1), argc(argc), argv(argv)
{
    // Setup the error handler to catch errors...
    // The default one exits, and makes it very hard to 
    // track down errors.  We need core dumps!
    XSetErrorHandler(x_error_handler);
    if(!tlock)
	tlock=scinew Mutex;
    Tcl_SetLock(do_lock, do_unlock);
    Tk_SetSelectProc((Tk_SelectProc*)Task::mtselect);
    tkMain(argc, argv);
    TCL::initialize();
}

TCLTask::~TCLTask()
{
}

int TCLTask::body(int)
{
    Tk_FirstPrompt();
    Tk_MainLoop();
    Tcl_Eval(the_interp, "exit");
    return 0;
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
