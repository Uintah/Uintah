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
 *  Modified for distributed Dataflow by:
 *   Michelle Miller
 *   May 1998
 *
 *  Copyright (C) 1999 U of U
 */

#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <Core/GuiInterface/GuiServer.h>
#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/RCParse.h>

#ifdef SCI_PARALLEL
#include <Core/CCA/Component/PIDL/PIDL.h>
#endif

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sys/stat.h>

#ifdef _WIN32
#include <afxwin.h>
#endif

#define VERSION "1.3.1" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml

using namespace SCIRun;

namespace SCIRun {
extern bool global_remote;
void set_guiManager( GuiManager * );
}

int global_argc;
char** global_argv;

namespace SCIRun {
extern env_map scirunrc;             // contents of .scirunrc
// these symbols live in Dataflow/Network/PackageDB.cc
extern string SCIRUN_SRCTOP;         // = INSTALL_DIR/SCIRun/src
extern string SCIRUN_OBJTOP;         // = BUILD_DIR
extern string DEFAULT_LOAD_PACKAGE;  // configured packages
}

#ifndef PSECORETCL
#error You must set PSECORETCL to the Dataflow/Tcl path
#endif

#ifndef SCICORETCL
#error You must set SCICORETCL to the Core/Tcl path
#endif

#ifndef DEF_LOAD_PACK
#error You must set a DEFAULT_PACKAGE_PATH or life is pretty dull
#endif

#ifndef ITCL_WIDGETS
#error You must set ITCL_WIDGETS to the iwidgets/scripts path
#endif

void
usage()
{
  cout << "Usage: scirun [args] [net_file]\n";
  cout << "       [-]-v[ersion] : prints out version information\n";
  cout << "       [-]-h[elp]    : prints usage information\n";
  cout << "       net_file      : SCIRun Network Input File\n";
  exit( 0 );
}

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???

void
parse_args( int argc, char *argv[] )
{
  for( int cnt = 0; cnt < argc; cnt++ )
    {
      string arg( argv[ cnt ] );
      if( ( arg == "--version" ) || ( arg == "-version" )
	  || ( arg == "-v" ) || ( arg == "--v" ) ){
	cout << "Version: " << VERSION << "\n";
	exit( 0 );
      } else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
		  ( arg == "-h" ) ||  ( arg == "--h" ) ) {
	usage();
      } else {
	  struct stat buf;
	  if (stat(arg.c_str(),&buf) < 0) {
	      cerr << "Couldn't find net file " << arg
		   << ".\nNo such file or directory.  Exiting." << endl;
	      exit(0);
	  }
      }
    }
}

int
main(int argc, char *argv[] )
{
  parse_args( argc, argv );

  global_argc=argc;
  global_argv=argv;

  // these symbols live in Dataflow/Network/PackageDB.cc but are reset here
  SCIRUN_SRCTOP = SRCTOP;
  SCIRUN_OBJTOP = OBJTOP;
  DEFAULT_LOAD_PACKAGE = DEF_LOAD_PACK;

#if 0
 ifdef SCI_PARALLEL
  try {
    PIDL::PIDL::initialize(argc, argv);
  } catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  } 
#endif

  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(argc, argv);
  Thread* t=new Thread(tcl_task, "TCL main event loop");
  t->detach();
  tcl_task->mainloop_waitstart();

  // Set up the TCL environment to find core components
  string result;
  TCL::eval("global PSECoreTCL CoreTCL",result);
  TCL::eval("set DataflowTCL "PSECORETCL,result);
  TCL::eval("set CoreTCL "SCICORETCL,result);
  TCL::eval("lappend auto_path "SCICORETCL,result);
  TCL::eval("lappend auto_path "PSECORETCL,result);
  TCL::eval("lappend auto_path "ITCL_WIDGETS,result);

  // Create initial network
  // We build the Network with a 1, indicating that this is the
  // first instantiation of the network class, and this network
  // should read the command line specified files (if any)
  Network* net=new Network(1);

  // Fork off task for the network editor.  It is a detached
  // task, and the Task* will be deleted by the task manager
  NetworkEditor* gui_task=new NetworkEditor(net);

  // Activate the network editor and scheduler.  Arguments and return
  // values are meaningless
  Thread* t2=new Thread(gui_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  char* HOME = getenv("HOME");
  
  if (HOME) {
    string home(HOME);
    home += "/.scirunrc";
    RCParse(home.c_str(),SCIRun::scirunrc);
  }

  // wait for the main window to display before continuing the startup.
  TCL::eval("tkwait visibility .top.globalViewFrame.canvas",result);


  // load the packages
  packageDB.loadPackage();
    

  // Now activate the TCL event loop
  tcl_task->release_mainloop();

#ifdef _WIN32
  // windows has a semantic problem with atexit(), so we wait here instead.
  HANDLE forever = CreateSemaphore(0,0,1,"forever");
  WaitForSingleObject(forever,INFINITE);
#endif

#ifndef __sgi
  Semaphore wait("main wait", 0);
  wait.down();
#endif
	
  // Never reached
  return 0;

}
