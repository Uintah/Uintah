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
#include <Core/Init/init.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Core/Comm/StringSocket.h>
#include <Core/Thread/Thread.h>

#include <Core/Services/ServiceLog.h>
#include <Core/Services/ServiceDB.h>
#include <Core/Services/ServiceManager.h>
#include <Core/SystemCall/SystemCallManager.h>

#include <sys/stat.h>
#include <fcntl.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using std::cout;

#ifdef _WIN32
#  include <windows.h>
#endif

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1209 
#endif


using namespace SCIRun;

void
usage()
{
  cout << "Usage: scirunremote [args]\n";
  cout << "    [-]-v[ersion]       : prints out version information\n";
  cout << "    [-]-h[elp]          : prints usage information\n";
  cout << "    [-]-p[ort] [PORT]   : start remote services port on port number PORT\n";
  exit( 0 );
}


// Parse the supported command-line arugments.
// Returns the argument # of the .net file

int parse_args( int argc, char *argv[] )
{
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
	else if ( ( arg == "--port" ) || ( arg == "-port" ) ||
	      ( arg == "-p" ) ||  ( arg == "--p" ) )
    {
      int port;
      if ((cnt+1 < argc) && string_to_int(argv[cnt+1], port)) 
      {
        if (port < 1024 || port > 65535) 
        {
            cerr << "Server port must be in range 1024-65535\n";
            exit(0);
        }
        cnt++;
      } 
      else 
      {
        port = 0;
      }
      sci_putenv("SCIRUN_SERVICE_PORT",to_string(port));
    }   
    cnt++;
  }
  return(0);
}



int
main(int argc, char *argv[], char **environment) {

  // Setup the SCIRun key/value environment
  create_sci_environment(environment, 0);
  sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);

  // Parse the command line arguments to find a network to execute
 parse_args( argc, argv );

  // The environment has been setup
  // Now split of a process for running external processes
  
  systemcallmanager_ = scinew SystemCallManager();
  systemcallmanager_->create();

  SCIRunInit();

  // Services start up... 

  // Create a database of all available services. The next piece of code
  // Scans both the SCIRun as well as the Packages directories to find
  // Services that need to be started. Services allow communication with
  // thirdparty software and are Threads that run asychronicly with
  // with the rest of SCIRun. Since the thirdparty software may be running
  // on a different platform it allows for connecting to remote machines
  // and running the service on a different machine 
 

  ServiceDBHandle servicedb = scinew ServiceDB;	

  servicedb->loadpackages();	// load all services and find all makers
  servicedb->activateall();		// activate all services

  
  // Services are started and created by the ServiceManager, which will be launched here
  // Two competing managers will be started, one for purely internal usage and one that
  // communicates over a socket. The latter will only be created if a port is set.
  // If the current instance of SCIRun should not provide any services to other instances
  // of SCIRun over the internet, the second manager will not be launched
  
  // A log file is not necessary but handy for debugging purposes
  //ServiceLogHandle internallogfile = scinew ServiceLog("scirun_internal_servicemanager.log");
  
  //IComAddress internaladdress("internal","servicemanager");
  //ServiceManager* internal_service_manager = scinew ServiceManager(servicedb,internaladdress,internallogfile); 
  //Thread* t_int = scinew Thread(internal_service_manager,"internal service manager",0,Thread::NotActivated);
  //t_int->setStackSize(1024*20);
  //t_int->activate(false);
  //t_int->detach();

  // Start an external service as well
  const char *serviceport_str = sci_getenv("SCIRUN_SERVICE_PORT");

  // Use the following environment setting is used to switch on IPv6 support
  // Most machines should be running a dual-host stack for the internet
  // connections, so it should not hurt to run in IPv6 mode. In most case
  // ipv4 address will work as well.
  //
  // It might be useful
  const char *serviceport_protocol = sci_getenv("SCIRUN_SERVICE_IPV6");
  std::string ipstr("");
  if (serviceport_protocol)
  {
    std::string protocol(serviceport_protocol);
	if ((protocol=="YES")||(protocol== "Y")||(protocol=="yes")||(protocol=="y")||(protocol=="1")||(protocol=="true")) ipstr = "ipv6";
  }
  
  if (serviceport_str)
  {
	// The protocol for conencting has been called "scirun"
	// In the near future this should be replaced with "sciruns" for
	// a secure version which will run over ssl. 
	
    // A log file is not necessary but handy for debugging purposes
    ServiceLogHandle externallogfile = scinew ServiceLog("scirun_external_servicemanager.log"); 
	
	IComAddress externaladdress("scirun","",serviceport_str,ipstr);
	ServiceManager* external_service_manager = scinew ServiceManager(servicedb,externaladdress,externallogfile); 
	Thread* t_ext = scinew Thread(external_service_manager,"external service manager",0,Thread::NotActivated);
	t_ext->setStackSize(1024*20);
    t_ext->setDaemon(true);
	t_ext->activate(false);
	t_ext->detach();
  }
  else
  {
    std::cout << "ERROR: No port specified for server" << std::endl;
  }
  
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "SCIRun remote server: running" << std::endl;
  servicedb->printservices();
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "type 'exit' or 'quit' to quit this program" << std::endl;


  // Wait for the user to type quit or exit  
  std::string input;
  while(1)
  {
    std::cin >> input;
    if (input.size() >= 4)
    {
        if (input.substr(0,4) == "exit") break;
        if (input.substr(0,4) == "quit") break;
        if (input.substr(0,4) == "Quit") break;
        if (input.substr(0,4) == "Exit") break;        
    }
  }
  
  // Kill this remote server
  Thread::exitAll(1);
  
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
#pragma reset woff 1424
#pragma reset woff 1209 
#endif
