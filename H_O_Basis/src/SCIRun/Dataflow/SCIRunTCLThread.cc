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

#include <SCIRun/Dataflow/SCIRunTCLThread.h>

#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Thread/Semaphore.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Util/soloader.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>

#include <iostream>
#include <tcl.h>
#include <tk.h>

typedef void (Tcl_LockProc)();

#ifdef _WIN32
#  ifdef __cplusplus
     extern "C" {
#  endif // __cplusplus
       __declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
       int tkMain(int argc, char** argv, 
          void (*nwait_func)(void*), void* nwait_func_data);
#  ifdef __cplusplus
     }
#  endif // __cplusplus

#else // _WIN32
  extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
  extern "C" int tkMain(int argc, char** argv,
            void (*nwait_func)(void*), void* nwait_func_data);

#endif // _WIN32

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

void wait(void* p);

void
do_lock3()
{
    TCLTask::lock();
}

void
do_unlock3()
{
    TCLTask::unlock();
}

void
wait(void* p)
{
    SCIRunTCLThread* thr = (SCIRunTCLThread*) p;
    thr->startTCL();
}

SCIRunTCLThread::SCIRunTCLThread(Network* net)
    : net(net), start("SCIRun startup semaphore", 0)
{
    Tcl_SetLock(do_lock3, do_unlock3);
}

void
SCIRunTCLThread::run()
{
    char* argv[2];
    argv[0] = "sr";
    argv[1] = 0;

    do_lock3();
    tkMain(1, argv, wait, this);
}

void
SCIRunTCLThread::startTCL()
{
    gui = new TCLInterface;
    new NetworkEditor(net, gui);

    packageDB->setGui(gui);
    gui->eval("set scirun2 1");
    gui->execute("wm withdraw .");

    start.up();
}

void
SCIRunTCLThread::tclWait()
{
    start.down();
}

}
