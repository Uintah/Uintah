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

#include <Dataflow/TCLThread/TCLThread.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <main/sci_version.h>
#include <tcl.h>
#include <tk.h>

typedef void (Tcl_LockProc)();

using namespace SCIRun;

#ifdef _WIN32
#  define SHARE __declspec(dllimport)
#  ifdef __cplusplus
     extern "C" {
#  endif // __cplusplus
#  ifndef EXPERIMENTAL_TCL_THREAD
       __declspec(dllimport) void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
#  endif
       int tkMain(int argc, char** argv, 
                  void (*nwait_func)(void*), void* nwait_func_data);
#  ifdef __cplusplus
     }
#  endif // __cplusplus
#else // not _WIN32
#  define SHARE
#  ifndef EXPERIMENTAL_TCL_THREAD
     extern "C" void Tcl_SetLock(Tcl_LockProc*, Tcl_LockProc*);
#  endif
     extern "C" int tkMain(int argc, char** argv,
                           void (*nwait_func)(void*), void* nwait_func_data);
#endif // _WIN32

extern "C" SHARE Tcl_Interp* the_interp;

#include <stdio.h>

using namespace std;


static void
wait(void* p)
{
  TCLThread* thr = (TCLThread*) p;
  thr->startNetworkEditor();
}


static void
do_lock2()
{
  TCLTask::lock();
}

static void
do_unlock2()
{
  TCLTask::unlock();
}


static
int
x_error_handler2(Display* dpy, XErrorEvent* error)
{
#ifndef _WIN32
    char msg[200];
    XGetErrorText(dpy, error->error_code, msg, 200);
    cerr << "X Error: " << msg << endl;
    abort();
#endif
    return 0; // Never reached...
}

TCLThread::TCLThread(int argc, char* argv[], Network* net, int startnetno) :
  argc(argc), argv(argv), net(net), startnetno(startnetno),
  cont("TCLThread startup continue semaphore", 0),
  start("TCLThread startup semaphore", 0)
{
  // Setup the error handler to catch errors...
  // The default one exits, and makes it very hard to 
  // track down errors.  We need core dumps!
  XSetErrorHandler(x_error_handler2);
#ifndef EXPERIMENTAL_TCL_THREAD
  Tcl_SetLock(do_lock2, do_unlock2);
#endif
}


void
TCLThread::run()
{
#ifndef EXPERIMENTAL_TCL_THREAD
  do_lock2();
#endif
  tkMain(1, argv, wait, this);
}


bool
TCLThread::check_for_newer_scirunrc() {
  const char *rcversion = sci_getenv("SCIRUN_RCFILE_VERSION");
  const string curversion = (rcversion ? rcversion : "");
  const string newversion = 
    string(SCIRUN_VERSION)+"."+string(SCIRUN_RCFILE_SUBVERSION);
  
  // If the .scirunrc is an old version
  if (curversion != newversion)
    // Ask them if they want to copy over a new one
    return gui->eval("promptUserToCopySCIRunrc") == "1";
  return false; // current version is equal to newest version
}
  

void
TCLThread::startNetworkEditor()
{
  gui = new TCLInterface;

  // We parse the scirunrc file here before creating the network
  // editor.  Note that this may fail, but we need the network editor
  // up before we can prompt the user what to do.  This means that the
  // environment variables used by the network editor are assumed to
  // be in the same state as the default ones in the srcdir/scirunrc
  // file.  For now only SCIRUN_NOGUI is affected.
  const bool scirunrc_parsed = find_and_parse_scirunrc();

  // Create the network editor here.  For now we just dangle it and
  // let exitAll destroy it with everything else at the end.
  if (sci_getenv_p("SCIRUN_NOGUI"))
  {
    gui->eval(string("rename unknown _old_unknown"));
    gui->eval(string("proc unknown args {\n") +
              string("    catch \"[uplevel 1 _old_unknown $args]\" result\n") +
              string("    return 0\n") +
              //string("    return $result\n") +
              string("}"));
  }
  new NetworkEditor(net, gui);

  // If the user doesnt have a .scirunrc file, or it is out of date,
  // provide them with a default one
  if (!scirunrc_parsed || (scirunrc_parsed && check_for_newer_scirunrc()))
  {  
    copy_and_parse_scirunrc();
  }

  // Determine if we are loading an app.
  const bool powerapp_p = (startnetno && ends_with(argv[startnetno],".app"));
  if (!powerapp_p)
  {
    gui->eval("set PowerApp 0");
    // Wait for the main window to display before continuing the startup.
    gui->eval("wm deiconify .");
    gui->eval("tkwait visibility $minicanvas");
    gui->eval("showProgress 1 0 1");
  }
  else
  { // If loading an app, don't wait.
    gui->eval("set PowerApp 1");
    if (argv[startnetno+1])
    {
      gui->eval("set PowerAppSession {"+string(argv[startnetno+1])+"}");
    }
    // Determine which standalone and set splash.
    if(strstr(argv[startnetno], "BioTensor"))
    {
      gui->eval("set splashImageFile $bioTensorSplashImageFile");
      gui->eval("showProgress 1 2575 1");
    }
    else if(strstr(argv[startnetno], "BioFEM"))
    {
      gui->eval("set splashImageFile $bioFEMSplashImageFile");
      gui->eval("showProgress 1 465 1");
    }
    else if(strstr(argv[startnetno], "BioImage"))
    {
      // Need to make a BioImage splash screen.
      gui->eval("set splashImageFile $bioImageSplashImageFile");
      gui->eval("showProgress 1 660 1");
    }
    else if(strstr(argv[startnetno], "FusionViewer"))
    {
      // Need to make a FusionViewer splash screen.
      gui->eval("set splashImageFile $fusionViewerSplashImageFile");
      gui->eval("showProgress 1 310 1");
    }
  }

  packageDB = new PackageDB(gui);
  // load the packages
  packageDB->loadPackage(!sci_getenv_p("SCIRUN_LOAD_MODULES_ON_STARTUP"));  

  if (!powerapp_p)
  {
    gui->eval("hideProgress");
  }
  
  // Check the dynamic compilation directory for validity
  sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->eval("activate_file_submenus");

  // Test for shaders.
  ShaderProgramARB::init_shaders_supported();

  // wait for main to release its semaphore
  mainloop_wait();

  // Load the Network file specified from the command line
  if (startnetno)
  {
    gui->eval("loadnet {"+string(argv[startnetno])+"}");
    if (sci_getenv_p("SCIRUN_EXECUTE_ON_STARTUP") || 
        sci_getenv_p("SCI_REGRESSION_TESTING"))
    {
      gui->eval("netedit scheduleall");
    }
  }
}


static
int
exitproc(ClientData, Tcl_Interp*, int, TCLCONST char* [])
{
  Thread::exitAll(0);
  return TCL_OK; // not reached
}


void
TCLThread::mainloop_wait()
{
  Tcl_CreateCommand(the_interp, "exit", exitproc, 0, 0);
#ifndef EXPERIMENTAL_TCL_THREAD
  do_unlock2();
#endif

  // The main program will want to know that we are started...
  start.up();

  // Wait for the main program to tell us that all initialization
  // has occurred...
  cont.down();

#ifdef EXPERIMENTAL_TCL_THREAD
  // windows doesn't communicate TCL with threads like other OSes do.
  // do instead of direct TCL communication, setup tcl callbacks
  TCLTask::setTCLEventCallback();
#else
  do_lock2();
#endif
}


void
TCLThread::mainloop_waitstart()
{
  start.down();
}


void
TCLThread::release_mainloop()
{
  cont.up();
}


