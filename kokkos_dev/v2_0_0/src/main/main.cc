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

#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/RCParse.h>
#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif

#include <sci_defs.h>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef _WIN32
#include <afxwin.h>
#endif

// This needs to be synced with the contents of
// SCIRun/doc/edition.xml and Dataflow/GUI/NetworkEditor.tcl
#define VERSION "1.20.3"

using namespace SCIRun;


namespace SCIRun {
extern env_map scirunrc;             // contents of .scirunrc
}

#ifndef PSECORETCL
#error You must set PSECORETCL to the Dataflow/Tcl path
#endif

#ifndef SCICORETCL
#error You must set SCICORETCL to the Core/Tcl path
#endif

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

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???

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
      cout << "Version: " << VERSION << "\n";
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
      putenv("SCI_REGRESSION_TESTING=1");
    }
    else if ( arg == "--nosplash" )
    {
      putenv("SCI_NOSPLASH=1");
    }
    else
    {
      struct stat buf;
      if (stat(arg.c_str(),&buf) < 0)
      {
	cerr << "Couldn't find net file " << arg
	     << ".\nNo such file or directory.  Exiting." << endl;
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


int
main(int argc, char *argv[] )
{
  const int startnetno = parse_args( argc, argv );

#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif

  // determine if we are loading an app
  char* app = 0;
  app = strstr(argv[startnetno],".app");

  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(1, argv);  // Discard argv on Tk side.
  Thread* t=new Thread(tcl_task, "TCL main event loop");
  t->detach();
  tcl_task->mainloop_waitstart();

  // Create user interface link
  GuiInterface* gui = new TCLInterface();

  // Set up the TCL environment to find core components
  string result;
  gui->eval("global PSECoreTCL CoreTCL",result);
  gui->eval("set DataflowTCL "PSECORETCL,result);
  gui->eval("set CoreTCL "SCICORETCL,result);
  gui->eval("set SCIRUN_SRCDIR "SCIRUN_SRCDIR,result);
  gui->eval("set SCIRUN_OBJDIR "SCIRUN_OBJDIR,result);
  gui->eval("lappend auto_path "SCICORETCL,result);
  gui->eval("lappend auto_path "PSECORETCL,result);
  gui->eval("lappend auto_path "ITCL_WIDGETS,result);
  gui->eval("global scirun2", result);
  gui->eval("set scirun2 0", result);

  // Create initial network
  packageDB = new PackageDB(gui);

  Network* net=new Network();

  Scheduler* sched_task=new Scheduler(net);

  new NetworkEditor(net, gui);

  // Activate the scheduler.  Arguments and return
  // values are meaningless
  Thread* t2=new Thread(sched_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  {
    ostringstream str;
    
    bool foundrc=false;
    str << "Parsing .scirunrc... ";
  
    // check the local directory
    foundrc = RCParse(".scirunrc",SCIRun::scirunrc);
    if (foundrc)
      str << "./.scirunrc" << endl;

    // check the BUILD_DIR
    if (!foundrc) {
      foundrc = RCParse((string(OBJTOP) + "/.scirunrc").c_str(),
			SCIRun::scirunrc);
      if (foundrc)
	str << OBJTOP << "/.scirunrc" << endl;
    }

    // check the user's home directory
    if (!foundrc) {
      char* HOME = getenv("HOME");
  
      if (HOME) {
	string home(HOME);
	home += "/.scirunrc";
	foundrc = RCParse(home.c_str(),SCIRun::scirunrc);
	if (foundrc)
	  str << home << endl;
      }
    }

    // check the INSTALL_DIR
    if (!foundrc) {
      foundrc = RCParse((string(SRCTOP) + "/.scirunrc").c_str(),
			SCIRun::scirunrc);
      if (foundrc)
	str << SRCTOP << "/.scirunrc" << endl;
    }

    // Since the dot file is optional report only if it was found.
    if( foundrc )
    {
      cout << str.str();
    }
    else
    {
      // check to make sure home directory is writeable.
      char* HOME = getenv("HOME");
      if (HOME)
      {
	string homerc = string(HOME) + "/.scirunrc";
	int fd;
	if ((fd = creat(homerc.c_str(), S_IREAD | S_IWRITE)) != -1)
	{
	  close(fd);
	  unlink(homerc.c_str());

	  string tclresult;
	  gui->eval("licenseDialog 1", result);
	  if (result == "cancel")
	  {
	    Thread::exitAll(1);
	  }
	  else if (result == "accept")
	  {
	    if ((fd = creat(homerc.c_str(), S_IREAD | S_IWRITE)) != -1)
	    {
	      close(fd);
	    }
	  }
	}
      }	  
    }
  }

  // set splash to be main one unless later changed due to a standalone
  packageDB->setSplashPath("main/scisplash.ppm");

  // wait for the main window to display before continuing the startup.
  // if loading an app, don't wait

  if(!app) {
    gui->eval("wm deiconify .", result);
    gui->eval("tkwait visibility .top.globalViewFrame.canvas",result);
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
  gui->eval("getOnTheFlyLibsDir",result);
  string envarstr = "SCIRUN_ON_THE_FLY_LIBS_DIR=" + result;
  char *envar = scinew char[envarstr.size()+1];
  memcpy(envar, envarstr.c_str(), envarstr.size());
  envar[envarstr.size()] = '\0';
  putenv(envar);

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->execute("activate_file_submenus");

  if (startnetno)
  {
    string command = string( "loadnet {" ) + argv[startnetno] + string("}");
    gui->eval(command.c_str(), result);

    if (execute_flag || getenv("SCI_REGRESSION_TESTING"))
    {
      gui->eval("netedit scheduleall", result);
    }
  }

  if (getenv("SCI_REGRESSION_TESTING"))
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
