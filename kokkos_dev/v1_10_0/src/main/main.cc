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
#include <sci_defs.h>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sys/stat.h>

#ifdef _WIN32
#include <afxwin.h>
#endif

#define VERSION "1.10.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml

using namespace SCIRun;


namespace SCIRun {
extern env_map scirunrc;             // contents of .scirunrc
extern bool regression_testing_flag;
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
  cout << "Usage: scirun [args] [net_file]\n";
  cout << "       [-]-r[egression] : regression test a network\n";
  cout << "       [-]-e[xecute]    : executes the given network on startup\n";
  cout << "       [-]-v[ersion]    : prints out version information\n";
  cout << "       [-]-h[elp]       : prints usage information\n";
  cout << "       net_file         : SCIRun Network Input File\n";
  exit( 0 );
}

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???

int
parse_args( int argc, char *argv[] )
{
  regression_testing_flag = false;

  int found = 0;
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
      regression_testing_flag = true;
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
      if (found)
      {
	usage();
      }
      found = cnt;
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

  bool foundrc=false;
  cout << "Parsing .scirunrc... ";
  
  // check the local directory
  foundrc = RCParse(".scirunrc",SCIRun::scirunrc);
  if (foundrc)
    cout << "./.scirunrc" << endl;

  // check the BUILD_DIR
  if (!foundrc) {
    foundrc = RCParse((string(OBJTOP) + "/.scirunrc").c_str(),
                      SCIRun::scirunrc);
    if (foundrc)
      cout << OBJTOP << "/.scirunrc" << endl;
  }

  // check the user's home directory
  if (!foundrc) {
    char* HOME = getenv("HOME");
  
    if (HOME) {
      string home(HOME);
      home += "/.scirunrc";
      foundrc = RCParse(home.c_str(),SCIRun::scirunrc);
      if (foundrc)
        cout << home << endl;
    }
  }

  // check the INSTALL_DIR
  if (!foundrc) {
    foundrc = RCParse((string(SRCTOP) + "/.scirunrc").c_str(),
                      SCIRun::scirunrc);
    if (foundrc)
      cout << SRCTOP << "/.scirunrc" << endl;
  }

  if( !foundrc )
    {
      cout << "not found.  (Note: This is not an error, though you\n"
	   << "might consider putting a .scirunrc file in your home dir.)\n";
    }

  // wait for the main window to display before continuing the startup.
  gui->eval("tkwait visibility .top.globalViewFrame.canvas",result);

  // load the packages
  packageDB->loadPackage();

  // Activate "File" menu sub-menus once packages are all loaded.
  gui->eval("activate_file_submenus",result);

  if (startnetno)
  {
    string command = string( "loadnet " ) + argv[startnetno];
    gui->eval(command.c_str(), result);

    if (execute_flag || regression_testing_flag)
    {
      gui->eval("ExecuteAll", result);
    }
  }

  if (regression_testing_flag)
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
