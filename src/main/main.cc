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
 *  main.cc: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1999 U of U
 */

#include <main/sci_version.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Core/Comm/StringSocket.h>
#include <Core/Thread/Thread.h>
#include <sys/stat.h>
#include <fcntl.h>

#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
   namespace SCIRun {
      extern void macImportExportForceLoad();
   }
#endif

#include <string>
#include <iostream>
using std::cout;

#ifdef _WIN32
#  include <afxwin.h>
#endif


using namespace SCIRun;

void
usage()
{
  cout << "Usage: scirun [args] [net_file] [session_file]\n";
  cout << "    [-]-r[egression]    : regression test a network\n";
  cout << "    [-]-s[erver] [PORT] : start a TCL server on port number PORT\n";
  cout << "    [-]-e[xecute]       : executes the given network on startup\n";
  cout << "    [-]-v[ersion]       : prints out version information\n";
  cout << "    [-]-h[elp]          : prints usage information\n";
  cout << "    [--nosplash]        : disable the splash screen\n";
  cout << "    net_file            : SCIRun Network Input File\n";
  cout << "    session_file        : PowerApp Session File\n";
  exit( 0 );
}


// Parse the supported command-line arugments.
// Returns the argument # of the .net file
int
parse_args( int argc, char *argv[] )
{
  int found = 0;
  bool powerapp = false;
  int cnt = 1;
  while (cnt < argc)
  {
    string arg( argv[ cnt ] );
    if( ( arg == "--version" ) || ( arg == "-version" )
	|| ( arg == "-v" ) || ( arg == "--v" ) )
    {
      cout << "Version: " << SCIRUN_VERSION << "\n";
      exit( 0 );
    }
    else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
	      ( arg == "-h" ) ||  ( arg == "--h" ) )
    {
      usage();
    }
    else if ( ( arg == "--execute" ) || ( arg == "-execute" ) ||
	      ( arg == "-e" ) ||  ( arg == "--e" ) )
    {
      sci_putenv("SCIRUN_EXECUTE_ON_STARTUP","1");
    }
    else if ( ( arg == "--regression" ) || ( arg == "-regression" ) ||
	      ( arg == "-r" ) ||  ( arg == "--r" ) )
    {
      sci_putenv("SCI_REGRESSION_TESTING","1");
    }
    else if ( arg == "--nosplash" )
    {
      sci_putenv("SCIRUN_NOSPLASH", "1");
    }
    else if ( ( arg == "--server" ) || ( arg == "-server" ) ||
	      ( arg == "-s" ) ||  ( arg == "--s" ) )
    {
      int port;
      if ((cnt+1 < argc) && string_to_int(argv[cnt+1], port)) {
	if (port < 1024 || port > 65535) {
	  cerr << "Server port must be in range 1024-65535\n";
	  exit(0);
	}
	cnt++;
      } else {
	port = 0;
      }
      sci_putenv("SCIRUN_SERVER_PORT",to_string(port));
    }    
    else
    {
      struct stat buf;
      if (stat(arg.c_str(),&buf) < 0)
      {
	std::cerr << "Couldn't find net file " << arg
		  << ".\nNo such file or directory.  Exiting." << std::endl;
	exit(0);
      }

      if (found && !powerapp)
      {
	usage();
      }

      // determine if it is a PowerApp
      if(strstr(arg.c_str(),".app")) {
	powerapp = true;
	found = cnt;
      }
      else if(!powerapp) {
	found = cnt;
      }
    }
    cnt++;
  }
  return found;
}


class RegressionKiller : public Runnable
{
public:
  void run()
  {
    int tmp, seconds = 300;
    const char *timeout = sci_getenv("SCIRUN_REGRESSION_TESTING_TIMEOUT");
    if (timeout && string_to_int(timeout, tmp)) {
      seconds = tmp;
    }
    sleep(seconds);
    cout << "\n";
    cout << "main.cc: RegressionKiller: Regression test timed out\n";
    cout << "         after " << seconds << " seconds.  Killing SCIRun.\n\n";
    Thread::exitAll(1);
  }
};



// show_licence_and_copy_sciunrc is not in Core/Util/Environment.h because it
// depends on GuiInterface to present the user with the license dialog.
void
show_license_and_copy_scirunrc(GuiInterface *gui) {
  const string tclresult = gui->eval("licenseDialog 1");
  if (tclresult == "cancel")
  {
    Thread::exitAll(1);
  }
  // check to make sure home directory is there
  const char* HOME = sci_getenv("HOME");
  const char* srcdir = sci_getenv("SCIRUN_SRCDIR");
  ASSERT(HOME);
  ASSERT(srcdir);
  if (!HOME) return;
  // If the user accepted the license then create a .scirunrc for them
  if (tclresult == "accept") {
    string homerc = string(HOME)+"/.scirunrc";
    string cmd = string("cp -f ")+srcdir+string("/scirunrc ")+homerc;
    std::cout << "Copying default " << srcdir << "/scirunrc to " <<
      homerc << "...\n";
    if (sci_system(cmd.c_str())) {
      std::cerr << "Error executing: " << cmd << std::endl;
    } else { 
      // if the scirunrc file was copied, then parse it
      parse_scirunrc(homerc);
    }
  }
}


class TCLSocketRunner : public Runnable
{
private:
  TCLInterface *gui_;
  StringSocket *transmitter_;
public:
  TCLSocketRunner(TCLInterface *gui, StringSocket *dt) : 
    gui_(gui), transmitter_(dt) {}
  void run()
  {
    string buffer;
    while (1) {
      buffer.append(transmitter_->getMessage());
      if (gui_->complete_command(buffer)) {
	buffer = gui_->eval(buffer);
	if (!buffer.empty()) buffer.append("\n");
	transmitter_->putMessage(buffer+"scirun> ");
	buffer.clear();
      } else {
	transmitter_->putMessage("scirun>> ");
      }
    }
  }
};


int
main(int argc, char *argv[], char **environment) {
  // Setup the SCIRun key/value environment
  create_sci_environment(environment, 0);
  sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);

  // Parse the command line arguments to find a network to execute
  const int startnetno = parse_args( argc, argv );

#if defined(__APPLE__)  
  macImportExportForceLoad(); // Attempting to force load (and thus
                              // instantiation of static constructors) 
  macForceLoad();             // of Core/Datatypes and Core/ImportExport.
#endif

  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(1, argv);// Only passes program name to TCL
  // We need to start the thread in the NotActivated state, so we can
  // change the stack size.  The 0 is a pointer to a ThreadGroup which
  // will default to the global thread group.
  Thread* t=new Thread(tcl_task,"TCL main event loop",0, Thread::NotActivated);
  t->setStackSize(1024*1024);
  // False here is stating that the tread was stopped or not.  Since
  // we have never started it the parameter should be false.
  t->activate(false);
  t->detach();
  tcl_task->mainloop_waitstart();

  // Create user interface link
  TCLInterface *gui = new TCLInterface();

  // TCL Socket
  int port;
  const char *port_str = sci_getenv("SCIRUN_SERVER_PORT");
  if (port_str && string_to_int(port_str, port)) {
    StringSocket *transmitter = scinew StringSocket(port);
    cerr << "URL: " << transmitter->getUrl() << std::endl;
    transmitter->run();
    TCLSocketRunner *socket_runner = scinew TCLSocketRunner(gui, transmitter);
    (new Thread(socket_runner, "TCL Socket"))->detach();
  }

  // Create initial network
  packageDB = new PackageDB(gui);
  Network* net=new Network();
  Scheduler* sched_task=new Scheduler(net);
  new NetworkEditor(net, gui);

  // If the user doesnt have a .scirunrc file, provide them with a default one
  if (!find_and_parse_scirunrc()) show_license_and_copy_scirunrc(gui);

  // Activate the scheduler.  Arguments and return values are meaningless
  Thread* t2=new Thread(sched_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  // determine if we are loading an app
  const bool loading_app_p = strstr(argv[startnetno],".app");
  if (!loading_app_p) {
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
      gui->eval("set splashImageFile $bioFEMSplashImageFile");
      gui->eval("showProgress 1 310 1");
    } else if(strstr(argv[startnetno], "FusionViewer")) {
      // need to make a FusionViewer splash screen
      gui->eval("set splashImageFile $fusionViewerSplashImageFile");
      gui->eval("showProgress 1 310 1");
    }

  }

  packageDB->loadPackage();  // load the packages

  if (!loading_app_p) {
    gui->eval("hideProgress");
  }
  
  // Check the dynamic compilation directory for validity
  sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->eval("activate_file_submenus");
  
  // Determine if SCIRun is in regression testing mode
  const bool doing_regressions = sci_getenv_p("SCI_REGRESSION_TESTING");

  // Load the Network file specified from the command line
  if (startnetno) {
    gui->eval("loadnet {"+string(argv[startnetno])+"}");
    if (sci_getenv_p("SCIRUN_EXECUTE_ON_STARTUP") || doing_regressions) {
      gui->eval("netedit scheduleall");
    }
  }

  // When doing regressions, make thread to kill ourselves after timeout
  if (doing_regressions) {
    RegressionKiller *kill = scinew RegressionKiller();
    Thread *tkill = scinew Thread(kill, "Kill a hung SCIRun");
    tkill->detach();
  }

  // Now activate the TCL event loop
  tcl_task->release_mainloop();

#ifdef _WIN32
  // windows has a semantic problem with atexit(), so we wait here instead.
  HANDLE forever = CreateSemaphore(0,0,1,"forever");
  WaitForSingleObject(forever,INFINITE);
#endif

#if !defined(__sgi)
  Semaphore wait("main wait", 0);
  wait.down();
#endif
	
  return 0;
}
