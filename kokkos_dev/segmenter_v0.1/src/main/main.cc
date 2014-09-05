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
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/NetworkIO.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/TCLThread/TCLThread.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Init/init.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Geom/ShaderProgramARB.h>

#include <Core/Services/ServiceLog.h>
#include <Core/Services/ServiceDB.h>
#include <Core/Services/ServiceManager.h>
#include <Core/SystemCall/SystemCallManager.h>

#include <TauProfilerForSCIRun.h>

#include <sci_defs/ptolemy_defs.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string>
#include <iostream>
using std::cout;

#ifdef _WIN32
#  include <windows.h>
#endif

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma set woff 1209
#  pragma set woff 1424
#endif

#ifdef HAVE_PTOLEMY_PACKAGE
#  include <Packages/Ptolemy/Core/Comm/PtolemyServer.h>
#endif

using namespace SCIRun;

void
usage()
{
  cout << "Usage: scirun [args] [net_file] [session_file]\n";
  cout << "    [-]-r[egression]    : regression test a network\n";
  cout << "    [-]-s[erver] [PORT] : start a TCL server on port number PORT\n";
  cout << "    [-]-e[xecute]       : executes the given network on startup\n";
  cout << "    [-]-c[onvert]       : converts a .net to a .srn network and exits\n";
  cout << "    [-]-v[ersion]       : prints out version information\n";
  cout << "    [-]-h[elp]          : prints usage information\n";
  cout << "    [-]-p[ort] [PORT]   : start remote services port on port number PORT\n";
#ifdef HAVE_PTOLEMY_PACKAGE
  cout << "    [-]-SPAs[erver]     : start the S.P.A. server thread\n";
#endif
  //  cout << "    [-]-eai             : enable external applications interface\n";
  cout << "    [--nosplash]        : disable the splash screen\n";
  cout << "    net_file            : SCIRun Network Input File\n";
  cout << "    session_file        : PowerApp Session File\n";
  exit( 0 );
}

static bool doing_convert_ = false;
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
    else if ( ( arg == "--convert" ) || ( arg == "-convert" ) ||
	      ( arg == "-c" ) ||  ( arg == "--c" ) )
    {
      sci_putenv("SCIRUN_CONVERT_NET_TO_SRN","1");
      doing_convert_ = true;
    }
    else if ( ( arg == "--eai" ) || ( arg == "-eai" ))
    {
      sci_putenv("SCIRUN_EXTERNAL_APPLICATION_INTERFACE","1");
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

    else if (ends_with(string_tolower(arg),".srn"))
    {
      NetworkIO::load_net(arg);
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
    else if ( ( arg == "--port" ) || ( arg == "-port" ) ||
	      ( arg == "-p" ) ||  ( arg == "--p" ) )
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
      sci_putenv("SCIRUN_SERVICE_PORT",to_string(port));
      sci_putenv("SCIRUN_EXTERNAL_APPLICATION_INTERFACE","1");
    }
#ifdef HAVE_PTOLEMY_PACKAGE
    else if ( ( arg == "--SPAserver" ) || ( arg == "-SPAserver" ) ||
	      ( arg == "-SPAs" ) || ( arg == "--SPAs" ) || ( arg == "-spas" ) )
    {
      sci_putenv("PTOLEMY_CLIENT_PORT","1");
    }
#endif   
    else
    {
      struct stat buf;
      if (stat(arg.c_str(),&buf) < 0)
      {
	std::cerr << "Couldn't find net file " << arg
		  << ".\nNo such file or directory.  Exiting." 
		  << std::endl;
	exit(0);
      }
      string filename(string_tolower(arg));
      if (!ends_with(filename,".net") && !ends_with(filename,".app"))
      {
	std::cerr << "Valid net files end with .srn, .app, " 
                  << "(or .net prior to v1.25.2) exiting." << std::endl;
	exit(0);
      }

      if (found && !powerapp)
      {
	usage();
      }

      // determine if it is a PowerApp
      if (ends_with(arg,".app")) {
	powerapp = true;
	found = cnt;
      } else if(!powerapp) {
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
    Time::waitFor((double)seconds);
    cout << "\n";
    cout << "main.cc: RegressionKiller: Regression test timed out\n";
    cout << "         after " << seconds << " seconds.  Killing SCIRun.\n\n";
    cout << "ERROR: KILL REQUIRED TO CONTINUE.";
    cout.flush();
    cerr.flush();
    Thread::exitAll(1);
  }
};

class ConvertKiller : public Runnable
{
public:
  void run()
  {
    double seconds = 1.0;
    while (! NetworkIO::done_writing()) {
      Time::waitFor(seconds);
    }
    Thread::yield();
    Time::waitFor(1.0);
    TCLTask::lock();
    Thread::exitAll(1);
  }
};




// Services start up... 
void
start_eai() {
  // Create a database of all available services. The next piece of code
  // Scans both the SCIRun as well as the Packages directories to find
  // Services that need to be started. Services allow communication with
  // thirdparty software and are Threads that run asychronicly with
  // with the rest of SCIRun. Since the thirdparty software may be running
  // on a different platform it allows for connecting to remote machines
  // and running the service on a different machine 
  ServiceDBHandle servicedb = scinew ServiceDB;     
  // load all services and find all makers
  servicedb->loadpackages();
  // activate all services
  servicedb->activateall();
  
  // Services are started and created by the ServiceManager, 
  // which will be launched here
  // Two competing managers will be started, 
  // one for purely internal usage and one that
  // communicates over a socket. 
  // The latter will only be created if a port is set.
  // If the current instance of SCIRun should not provide any services 
  // to other instances of SCIRun over the internet, 
  // the second manager will not be launched
  
  const char *chome = sci_getenv("HOME");
  string scidir("");
  if (chome)
    scidir = chome+string("/SCIRun/");

  // A log file is not necessary but handy for debugging purposes
  ServiceLogHandle internallogfile = 
    scinew ServiceLog(scidir+"scirun_internal_servicemanager.log");
  
  IComAddress internaladdress("internal","servicemanager");
  ServiceManager* internal_service_manager = 
    scinew ServiceManager(servicedb, internaladdress, internallogfile); 
  Thread* t_int = 
    scinew Thread(internal_service_manager, "internal service manager",
		  0, Thread::NotActivated);
  t_int->setStackSize(1024*20);
  t_int->activate(false);
  t_int->detach();
  
  
  // Use the following environment setting to switch on IPv6 support
  // Most machines should be running a dual-host stack for the internet
  // connections, so it should not hurt to run in IPv6 mode. In most case
  // ipv4 address will work as well.
  // It might be useful
  std::string ipstr(sci_getenv_p("SCIRUN_SERVICE_IPV6")?"ipv6":"");
  
  // Start an external service as well
  const char *serviceport_str = sci_getenv("SCIRUN_SERVICE_PORT");
  // If its not set in the env, we're done
  if (!serviceport_str) return;
  
  // The protocol for conencting has been called "scirun"
  // In the near future this should be replaced with "sciruns" for
  // a secure version which will run over ssl. 
  
  // A log file is not necessary but handy for debugging purposes
  ServiceLogHandle externallogfile = 
    scinew ServiceLog(scidir+"scirun_external_servicemanager.log"); 
  
  IComAddress externaladdress("scirun","",serviceport_str,ipstr);
  ServiceManager* external_service_manager = 
    scinew ServiceManager(servicedb,externaladdress,externallogfile); 
  Thread* t_ext = 
    scinew Thread(external_service_manager,"external service manager",
		  0, Thread::NotActivated);
  t_ext->setStackSize(1024*20);
  t_ext->activate(false);
  t_ext->detach();
}  




int
main(int argc, char *argv[], char **environment) {

  TAU_PROFILE("main", "", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);

  // Setup the SCIRun key/value environment
  create_sci_environment(environment, 0);
  sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);
  sci_putenv("SCIRUN_RCFILE_SUBVERSION", SCIRUN_RCFILE_SUBVERSION);

  // Parse the command line arguments to find a network to execute
  const int startnetno = parse_args( argc, argv );

  SCIRunInit();

  // Now split off a process for running external processes
  systemcallmanager_ = scinew SystemCallManager();
  systemcallmanager_->create();
  start_eai();
  
  Network* net=new Network();

  // Activate the scheduler.  Arguments and return values are meaningless
  Thread* scheduler=new Thread(new Scheduler(net), "Scheduler");
  scheduler->setDaemon(true);
  scheduler->detach();

  // Start the TCL thread, takes care of loading packages and networks
  Thread* tcl=new Thread(new TCLThread(argc, argv, net, startnetno),
                         "TCL main event loop",0,Thread::Activated,1024*1024);
  tcl->detach();
        
  // When doing regressions, make thread to kill ourselves after timeout
  if (sci_getenv_p("SCI_REGRESSION_TESTING")) {
    RegressionKiller *kill = scinew RegressionKiller();
    Thread *tkill = scinew Thread(kill, "Kill a hung SCIRun");
    tkill->detach();
  }


#ifdef HAVE_PTOLEMY_PACKAGE
  //start the Ptolemy/spa server socket
  const char *pt_str = sci_getenv("PTOLEMY_CLIENT_PORT");
  if (pt_str && string_to_int(pt_str, port)) {
    cerr << "Starting SPA server thread" << std::endl;
    PtolemyServer *ptserv = new PtolemyServer(gui,net);
    Thread *pt = new Thread(ptserv, "Ptolemy/SPA Server", 0,
                            Thread::Activated, 1024*1024);
    pt->detach();
    PtolemyServer::servSem().up();
  }
#endif

  if (doing_convert_) {
    ConvertKiller *kill = scinew ConvertKiller();
    Thread *tkill = scinew Thread(kill, "Exit post convert net");
    tkill->detach();
  }

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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1209
#pragma reset woff 1424
#endif

