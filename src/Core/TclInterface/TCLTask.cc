//static char *id="@(#) $Id$";
 
/*
 *  TCLTask.cc:  Handle TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Exceptions/Exceptions.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Thread.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdlib.h>
#include <tcl.h>
#include <tk.h>

typedef void (Tcl_LockProc)();
typedef int (IsTclThreadProc)();

#ifdef _WIN32
#undef ASSERT
#include <afxwin.h>
#define GLXContext HGLRC
#ifdef __cplusplus
extern "C" {
#endif
__declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
__declspec(dllimport) void Tcl_SetIsTclThread(IsTclThreadProc* proc);
int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);
#ifdef __cplusplus
}
#endif

#else

extern "C" int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);
extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
extern "C" void Tcl_SetIsTclThread(IsTclThreadProc* proc);

#endif

extern "C" Tcl_Interp* the_interp;

namespace SCICore {
namespace TclInterface {

    using SCICore::Thread::Mutex;
    using SCICore::Thread::Thread;

static Mutex tlock("TCL task lock");
static Thread* owner;
static int lock_count;
static Thread* tcl_task_id;

static void do_lock()
{
  ASSERT(Thread::self() != 0);
    if(owner == Thread::self()){
      lock_count++;
//      cerr << "Recursively locked, count=" << lock_count << endl;
	return;
    }
    tlock.lock();
    lock_count=1;
    owner=Thread::self();
//    cerr << "Locked: owner=" << owner << ", count=" << lock_count << endl;
}

static void do_unlock()
{
    ASSERT(lock_count>0);
//    cerr << "Self=" << Task::self() << ", owner=" << owner << endl;
    ASSERT(Thread::self() == owner);
    if(--lock_count == 0){
	owner=0;
//	cerr << "Unlocked, count=" << lock_count << ", owner=" << owner << ", self=" << Task::self() << endl;
	tlock.unlock();
    } else {
//      cerr << "Recursively unlocked, count=" << lock_count << endl;
    }
}

static int is_tcl_thread()
{
  return Thread::self() == tcl_task_id;
}

static int x_error_handler(Display* dpy, XErrorEvent* error)
{
#ifndef _WIN32
    char msg[200];
    XGetErrorText(dpy, error->error_code, msg, 200);
    cerr << "X Error: " << msg << endl;
    abort();
#endif
    return 0; // Never reached...
}

static int exitproc(ClientData, Tcl_Interp*, int, char* [])
{
	printf("exitproc() {%s,%d}\n",__FILE__,__LINE__);
    Thread::exitAll(0);
    return TCL_OK; // not reached
}

TCLTask::TCLTask(int argc, char* argv[])
: cont("TCLTask startup continue semaphore", 0),
  start("TCLTask startup semaphore", 0),
  argc(argc), argv(argv)
{
    // Setup the error handler to catch errors...
    // The default one exits, and makes it very hard to 
    // track down errors.  We need core dumps!
    XSetErrorHandler(x_error_handler);

    Tcl_SetLock(do_lock, do_unlock);
    Tcl_SetIsTclThread(is_tcl_thread);
}

TCLTask::~TCLTask()
{
}

void wait_func(void* thatp)
{
    TCLTask* that=(TCLTask*)thatp;
    that->mainloop_wait();
}

void
TCLTask::run()
{
    tcl_task_id=Thread::self();

    // Acquire the lock before we go into the Tcl/Tk main loop.
    // From now on, it will only get unlocked when the GUI blocks.
    do_lock();

    tkMain(argc, argv, wait_func, (void*)this);
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
    do_unlock();

    // The main program will want to know that we are started...
    start.up();

    // Wait for the main program to tell us that all initialization
    // has occurred...
    cont.down();
    do_lock();
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
    if(owner == Thread::self()){
	lock_count++;
	return 1;
    }
    if(tlock.tryLock()){
	lock_count=1;
	owner=Thread::self();
//	cerr << "Locked (try): owner=" << owner << ", count=" << lock_count << endl;
	return 1;
    } else {
	cerr << "Try Lock failed" << endl;
	return 0;
    }
}

Thread* TCLTask::get_owner()
{
    return owner;
}

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/10/07 02:08:03  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:26:55  sparker
// Various #include cleanups
//
// Revision 1.4  1999/09/04 06:01:55  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.3  1999/08/28 17:54:52  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:45  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:16  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
