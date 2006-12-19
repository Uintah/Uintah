/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <SCIRun/Dataflow/SCIRunTCLThread.h>

#include <Dataflow/GuiInterface/TCLInterface.h>
#include <Dataflow/GuiInterface/TCLTask.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>

#include <iostream>
#include <tcl.h>
#include <tk.h>

typedef void (Tcl_LockProc)();

// #ifdef _WIN32
// #  ifdef __cplusplus
//      extern "C" {
// #  endif // __cplusplus
//        __declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
//        int tkMain(int argc, char** argv, 
//           void (*nwait_func)(void*), void* nwait_func_data);
// #  ifdef __cplusplus
//      }
// #  endif // __cplusplus

// #else // _WIN32
//   extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
//   extern "C" int tkMain(int argc, char** argv,
//             void (*nwait_func)(void*), void* nwait_func_data);

// #endif // _WIN32

#ifdef _WIN32
#  define EXPERIMENTAL_TCL_THREAD
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#endif // _WIN32


extern "C" SCISHARE Tcl_Interp* the_interp;

#ifndef EXPERIMENTAL_TCL_THREAD
extern "C" SCISHARE void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
#endif

namespace SCIRun {

SCIRunTCLThread *SCIRunTCLThread::init_ptr_ = 0;

void
do_lock3()
{
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
#endif
}

void
do_unlock3()
{
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::unlock();
#endif
}

int
SCIRunTCLThread::wait(Tcl_Interp *interp)
{
  the_interp = interp;
  ASSERT(init_ptr_);
  return init_ptr_->startTCL();
}

SCIRunTCLThread::SCIRunTCLThread(Network* net)
  : net(net), start("SCIRun startup semaphore", 0)
{
#ifndef EXPERIMENTAL_TCL_THREAD
  Tcl_SetLock(do_lock3, do_unlock3);
#endif
}

void
SCIRunTCLThread::run()
{
  char* argv[2];
  argv[0] = "sr";
  argv[1] = 0;

  do_lock3();
  init_ptr_ = this;

  if (sci_getenv_p("SCIRUN_NOGUI")) {
    Tcl_Main(1, argv, wait);
  } else {
    Tk_Main(1, argv, wait);
  }
}

int
SCIRunTCLThread::startTCL()
{
  gui = new TCLInterface;
  new NetworkEditor(net, gui);
  // TODO: scirunrc file handling

  // Find and set the on-the-fly directory
  sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

  packageDB->setGui(gui);
  gui->eval("set scirun2 1");
  gui->execute("wm withdraw .");

  start.up();
  return TCL_OK;
}

void
SCIRunTCLThread::tclWait()
{
  start.down();
}

}
