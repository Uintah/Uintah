/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

 
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

#include <Core/GuiInterface/TCLTask.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdlib.h>
#include <tcl.h>
#include <tk.h>

typedef void (Tcl_LockProc)();
//typedef int (IsTclThreadProc)();

#ifdef _WIN32
#undef ASSERT
#include <afxwin.h>
#define GLXContext HGLRC
#ifdef __cplusplus
extern "C" {
#endif
__declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
  //__declspec(dllimport) void Tcl_SetIsTclThread(IsTclThreadProc* proc);
int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);
#ifdef __cplusplus
}
#endif

#else

extern "C" int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);
extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
//extern "C" void Tcl_SetIsTclThread(IsTclThreadProc* proc);

#endif

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {


static Mutex tlock("TCL task lock");
static Thread* owner;
static int lock_count;
//static Thread* tcl_task_id;

static void do_lock()
{
    ASSERT(Thread::self() != 0);
    if(owner == Thread::self()){
      lock_count++;
      return;
    }
    tlock.lock();
    lock_count=1;
    owner=Thread::self();
}

static void do_unlock()
{
    ASSERT(lock_count>0);
    ASSERT(Thread::self() == owner);
    if(--lock_count == 0){
	owner=0;
	tlock.unlock();
    } else {
    }
}

// static int is_tcl_thread()
// {
//   return Thread::self() == tcl_task_id;
// }

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
  //printf("exitproc() {%s,%d}\n",__FILE__,__LINE__);
  Thread::exitAll(0);
  return TCL_OK; // not reached
}

TCLTask::TCLTask(int argc, char* argv[])
  : argc(argc), argv(argv),
    cont("TCLTask startup continue semaphore", 0),
    start("TCLTask startup semaphore", 0)
{
    // Setup the error handler to catch errors...
    // The default one exits, and makes it very hard to 
    // track down errors.  We need core dumps!
    XSetErrorHandler(x_error_handler);

    Tcl_SetLock(do_lock, do_unlock);
    //    Tcl_SetIsTclThread(is_tcl_thread);
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
    //tcl_task_id=Thread::self();

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
	return 1;
    } else {
	return 0;
    }
}

Thread* TCLTask::get_owner()
{
    return owner;
}

} // End namespace SCIRun

