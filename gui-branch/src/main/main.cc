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

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sys/stat.h>

#ifdef _WIN32
#include <afxwin.h>
#endif

#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>

#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Resources/Resources.h>

#include <UI/tcltk/GuiInterface/NetworkEditor.h>
#include <UI/tcltk/GuiInterface/GuiManager.h>

#ifdef SCI_PARALLEL
#include <Core/CCA/Component/PIDL/PIDL.h>
#endif

#define VERSION "1.3.1" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml

using namespace SCIRun;

int global_argc;
char** global_argv;

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
  
  resources.read("../src/scirun.xml");
  
#ifdef SCI_PARALLEL
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
  

  gm = new TcltkManager;
  Network* net=new Network(1);
  Scheduler *scheduler = new Scheduler(net);
  
  NetworkEditor *gui = new NetworkEditor(net, scheduler,argc, argv);

  Thread* t2=new Thread(scheduler, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  gui->start();

  // exit

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
