/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#ifdef _WIN32
#include <windows.h>
#define GLXContext HGLRC
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#ifndef EXPERIMENTAL_TCL_THREAD
  __declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
#endif
int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);
#ifdef __cplusplus
}
#endif // __cplusplus

#else // _WIN32

#ifndef EXPERIMENTAL_TCL_THREAD
  extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
#endif
extern "C" int tkMain(int argc, char** argv, void (*nwait_func)(void*), void* nwait_func_data);


#endif // _WIN32

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {


static Mutex tlock("TCL task lock");
static Thread* owner;
static int lock_count;

static void do_lock()
{
    ASSERT(Thread::self() != 0);
    if(owner == Thread::self()){
      lock_count++;
      return;
    }
    bool blah = false;
    if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") != 0 && owner) {
      blah = true;
      //cerr << " Bef lock " << Thread::self()->getThreadName() << ": owned by " << owner->getThreadName() << endl;
    }
    tlock.lock();
    if (blah) {
      //cerr << " Got lock " << Thread::self()->getThreadName() << endl;
    }
    lock_count=1;
    owner=Thread::self();
    //if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") != 0)
      //cerr << "Locked by " << owner->getThreadName() << endl;
}

static void do_unlock()
{
    ASSERT(lock_count>0);
    ASSERT(Thread::self() == owner);
    if(--lock_count == 0){
	owner=0;
	tlock.unlock();
    //if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") != 0)
          //cerr << "UnLocked by " << Thread::self()->getThreadName() << endl;
    } else {
    }
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

static int exitproc(ClientData, Tcl_Interp*, int, TCLCONST char* [])
{

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

#ifndef EXPERIMENTAL_TCL_THREAD
    Tcl_SetLock(do_lock, do_unlock);
#endif
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

#ifdef EXPERIMENTAL_TCL_THREAD
// these two defined in TCLInterface.cc
void eventCheck(ClientData cd, int flags);
void eventSetup(ClientData cd, int flags);
// defined here so we can define EXPERIMENTAL_TCL_THREAD in only one place
void TCLTask::setTCLEventCallback()
{
  Tcl_CreateEventSource(eventSetup, eventCheck, 0);
}
#endif

Thread* TCLTask::get_owner()
{
    return owner;
}

} // End namespace SCIRun

