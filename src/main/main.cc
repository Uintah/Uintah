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

#include <sci_defs.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>
#include <sys/stat.h>
#include <fcntl.h>

#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif

#include <string>
#include <iostream>
using std::cout;

#ifdef _WIN32
#include <afxwin.h>
#endif

// This needs to be synced with the contents of
// SCIRun/doc/edition.xml
#define SCIRUN_VERSION "1.22.0"

using namespace SCIRun;

#ifndef LOAD_PACKAGE
#error You must set a LOAD_PACKAGE or life is pretty dull
#endif

#ifndef ITCL_WIDGETS
#error You must set ITCL_WIDGETS to the iwidgets/scripts path
#endif

static bool execute_flag = false;

void
usage()
{
  cout << "Usage: scirun [args] [net_file] [session_file]\n";
  cout << "       [-]-r[egression] : regression test a network\n";
  cout << "       [-]-e[xecute]    : executes the given network on startup\n";
  cout << "       [-]-v[ersion]    : prints out version information\n";
  cout << "       [-]-h[elp]       : prints usage information\n";
  cout << "       [--nosplash]     : disable the splash screen\n";
  cout << "       net_file         : SCIRun Network Input File\n";
  cout << "       session_file     : PowerApp Session File\n";
  exit( 0 );
}

// Parse the supported command-line arugments.
// Returns the argument # of the .net file
int
parse_args( int argc, char *argv[] )
{
  int found = 0;
  bool powerapp = false;
  for( int cnt = 1; cnt < argc; cnt++ )
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
      execute_flag = true;
    }
    else if ( ( arg == "--regression" ) || ( arg == "-regression" ) ||
	      ( arg == "-r" ) ||  ( arg == "--r" ) )
    {
      sci_putenv("SCI_REGRESSION_TESTING","1");
    }
    else if ( arg == "--nosplash" )
    {
      sci_putenv("SCI_NOSPLASH", "1");
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
  }
  return found;
}


class RegressionKiller : public Runnable
{
public:
  void run()
  {
    sleep(300);
    cout << "Regression test timeout, killing SCIRun.\n";
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
  if (!HOME) return;
  // If the user accepted the license then create a .scirunrc for them
  if (tclresult == "accept") {
    string homerc = string(HOME)+"/.scirunrc";
    string cmd = string("cp -f ")+SCIRUN_SRCDIR+string("/scirunrc ")+homerc;
    std::cout << "Copying default " << SCIRUN_SRCDIR << "/scirunrc to " <<
      homerc << "...\n";
    if (sci_system(cmd.c_str())) {      
      std::cerr << "Error executing: " << cmd << std::endl;
    } else { 
      // if the scirunrc file was copied, then parse it
      parse_scirunrc(homerc);
    }
  }
}



int
main(int argc, char *argv[], char **environment) {
  // Setup the SCIRun key/value environment
  create_sci_environment(environment);
  sci_putenv("SCIRUN_SRCDIR", SCIRUN_SRCDIR);
  sci_putenv("SCIRUN_OBJDIR", SCIRUN_OBJDIR);
  sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);

  // Parse the command line arguments to find a network to execute
  const int startnetno = parse_args( argc, argv );

#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
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
  GuiInterface* gui = new TCLInterface();
  // setup TCL auto_path to find core components
  gui->execute("lappend auto_path "SCIRUN_SRCDIR"/Core/GUI "
	       SCIRUN_SRCDIR"/Dataflow/GUI "ITCL_WIDGETS);
  gui->execute("set scirun2 0");

  // Create initial network
  packageDB = new PackageDB(gui);
  Network* net=new Network();
  Scheduler* sched_task=new Scheduler(net);
  new NetworkEditor(net, gui);

  // If the user doesnt have a .scirunrc file, provide them with a default one
  if (!find_and_parse_scirunrc()) show_license_and_copy_scirunrc(gui);

  // Activate the scheduler.  Arguments and return
  // values are meaningless
  Thread* t2=new Thread(sched_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  // set splash to be main one unless later changed due to a standalone
  packageDB->setSplashPath("main/scisplash.ppm");

  // determine if we are loading an app
  if(!strstr(argv[startnetno],".app")) {
    // wait for the main window to display before continuing the startup.
    // if loading an app, don't wait    
    gui->execute("wm deiconify .");
    gui->execute("tkwait visibility $minicanvas");
  } else {
    // set that we are loading an app and set the session file if provided
    if((startnetno + 1) < argc) {
      packageDB->setLoadingApp(true, argv[startnetno+1]);
    }
    else {
      packageDB->setLoadingApp(true);
    }

    // determine which standalone and set splash
    if(strstr(argv[startnetno], "BioTensor")) {
      packageDB->setSplashPath("Packages/Teem/Dataflow/GUI/splash-tensor.ppm");
    }   
  }

  // load the packages
  packageDB->loadPackage();
  
  // Check the dynamic compilation directory for validity
  sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->execute("activate_file_submenus");

  const bool doing_regressions = sci_getenv_p("SCI_REGRESSION_TESTING");

  if (startnetno)
  {
    gui->execute("loadnet {"+string(argv[startnetno])+"}");
    if (execute_flag || doing_regressions )
    {
      gui->execute("netedit scheduleall");
    }
  }

  if( doing_regressions )
  {
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
