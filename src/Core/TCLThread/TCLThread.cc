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

#include <Core/TCLThread/TCLThread.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/GuiInterface/TCLTask.h>
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

#include <stdio.h>

using namespace std;

void wait(void* p);

void
do_lock2()
{
  TCLTask::lock();
}

void
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

  Tcl_SetLock(do_lock2, do_unlock2);
}

void
TCLThread::run()
{
  do_lock2();
  tkMain(1, argv, wait, this);
}

// show_licence_and_copy_sciunrc is not in Core/Util/Environment.h because it
// depends on GuiInterface to present the user with the license dialog.
void
show_license_and_copy_scirunrc(GuiInterface *gui) {
  const string tclresult = gui->eval("licenseDialog 1");
  if (tclresult == "cancel") {
    Thread::exitAll(1);
  }
  // check to make sure home directory is there
  const char* HOME = sci_getenv("HOME");
  const char* srcdir = sci_getenv("SCIRUN_SRCDIR");
  const char* temp_rcfile_version = sci_getenv("SCIRUN_RCFILE_VERSION");
  string SCIRUN_RCFILE_VERSION;

  // If the .scirunrc file does not have a SCIRUN_RCFILE_VERSION variable...
  if( temp_rcfile_version == NULL ) {
    SCIRUN_RCFILE_VERSION = "bak";
  } else {
    SCIRUN_RCFILE_VERSION = temp_rcfile_version;
  }

  ASSERT(HOME);
  ASSERT(srcdir);
  if (!HOME) return;
  // If the user accepted the license then create a .scirunrc for them
  if (tclresult == "accept") {
    string homerc = string(HOME)+"/.scirunrc";
    string cmd;
    if (gui->eval("validFile "+homerc) == "1") {
      string backuprc = homerc + "." + SCIRUN_RCFILE_VERSION;
      cmd = string("cp -f ")+homerc+" "+backuprc;
      std::cout << "Backing up " << homerc << " to " << backuprc << std::endl;
      if (sci_system(cmd.c_str())) {
        std::cerr << "Error executing: " << cmd << std::endl;
      }
    }

    cmd = string("cp -f ")+srcdir+string("/scirunrc ")+homerc;
    std::cout << "Copying " << srcdir << "/scirunrc to " <<
      homerc << "...\n";
    if (sci_system(cmd.c_str())) {
      std::cerr << "Error executing: " << cmd << std::endl;
    } else { 
      // if the scirunrc file was copied, then parse it
      parse_scirunrc(homerc);
    }
  }
}

void
TCLThread::startNetworkEditor()
{
  gui = new TCLInterface;
  new NetworkEditor(net, gui);

  // If the user doesnt have a .scirunrc file, provide them with a default one
  if (!find_and_parse_scirunrc()) 
    show_license_and_copy_scirunrc(gui);
  else {
    const char *rcversion = sci_getenv("SCIRUN_RCFILE_VERSION");
    const string ver = 
      string(SCIRUN_VERSION)+"."+string(SCIRUN_RCFILE_SUBVERSION);
    // If the .scirunrc is an old version
    if (!rcversion || string(rcversion) != ver)
      // Ask them if they want to copy over a new one
      if (gui->eval("promptUserToCopySCIRunrc") == "1")
        show_license_and_copy_scirunrc(gui);
  }

  // determine if we are loading an app
  const bool powerapp_p = (startnetno && ends_with(argv[startnetno],".app"));
  if (!powerapp_p) {
    gui->eval("set PowerApp 0");
    // wait for the main window to display before continuing the startup.
    gui->eval("wm deiconify .");
    gui->eval("tkwait visibility $minicanvas");
    gui->eval("showProgress 1 0 1");
  } else { // if loading an app, don't wait
    gui->eval("set PowerApp 1");
    if (argv[startnetno+1]) {
      gui->eval("set PowerAppSession {"+string(argv[startnetno+1])+"}");
    }
    // determine which standalone and set splash
    if(strstr(argv[startnetno], "BioTensor")) {
      gui->eval("set splashImageFile $bioTensorSplashImageFile");
      gui->eval("showProgress 1 2575 1");
    } else if(strstr(argv[startnetno], "BioFEM")) {
      gui->eval("set splashImageFile $bioFEMSplashImageFile");
      gui->eval("showProgress 1 465 1");
    } else if(strstr(argv[startnetno], "BioImage")) {
      // need to make a BioImage splash screen
      gui->eval("set splashImageFile $bioImageSplashImageFile");
      gui->eval("showProgress 1 660 1");
    } else if(strstr(argv[startnetno], "FusionViewer")) {
      // need to make a FusionViewer splash screen
      gui->eval("set splashImageFile $fusionViewerSplashImageFile");
      gui->eval("showProgress 1 310 1");
    }

  }

  packageDB = new PackageDB(gui);
  packageDB->loadPackage();  // load the packages

  if (!powerapp_p) {
    gui->eval("hideProgress");
  }
  
  // Check the dynamic compilation directory for validity
  sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->eval("activate_file_submenus");

  // wait for main to release its semaphore
  mainloop_wait();

  // Load the Network file specified from the command line
  if (startnetno) {
    gui->eval("loadnet {"+string(argv[startnetno])+"}");
    if (sci_getenv_p("SCIRUN_EXECUTE_ON_STARTUP") || 
	sci_getenv_p("SCI_REGRESSION_TESTING")) {
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
  do_unlock2();

  // The main program will want to know that we are started...
  start.up();

  // Wait for the main program to tell us that all initialization
  // has occurred...
  cont.down();
  do_lock2();
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

void
wait(void* p)
{
  TCLThread* thr = (TCLThread*) p;
  thr->startNetworkEditor();
}
