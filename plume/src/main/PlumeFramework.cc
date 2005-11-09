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
 *  PlumeFramework.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Intitute
 *   University of Utah
 *   October 2005
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Util/Environment.h>

#include <Core/Thread/Thread.h>

#include <Core/CCA/spec/sci_sidl.h>

#include <SCIRun/Plume/PlumeFrameworkImpl.h>
#include <SCIRun/Core/PropertiesImpl.h>
#include <Core/Thread/Thread.h>
#include <main/SimpleManager.h>

#include <iostream>

using namespace SCIRun;
using namespace sci::cca;
using namespace sci::cca::ports;
using namespace sci::cca::core;
using namespace sci::cca::core::ports;
using namespace sci::cca::distributed;

#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml

void init();
void usage();
void parse_args( int argc, char *argv[]);
void add_args( const CoreFramework::pointer &framework, int argc, char *argv[]);

static std::string attach_url = "";
static std::string parent_url = "";
static std::string app_class = "";
static bool server = false;

int
main(int argc, char *argv[]) 
{
  parse_args(argc, argv);
  init();
  
  // Create a new framework
  try {
    DistributedFramework::pointer framework;

    if ( attach_url != "") 
      framework = pidl_cast<DistributedFramework::pointer>(PIDL::objectFrom(URL(attach_url)));
    else {
      DistributedFramework::pointer parent;
      if ( parent_url != "" ) parent = pidl_cast<DistributedFramework::pointer>(PIDL::objectFrom(URL(parent_url)));
      framework = new PlumeFrameworkImpl(parent);

      // add argv to local framework properties 
      add_args( framework, argc, argv);
    }

    if ( framework.isNull() ) {
      std::cerr << "Can not create/attach framework\n";
      return 1;
    }

    std::cerr << "Framework id: " << framework->getFrameworkID()->getString() << "\n";

    if ( app_class != "") {
      SimpleManager(framework, app_class);
    }
    if ( !server ) {
      framework->shutdownFramework();
      framework = 0;
    }

    //broadcast, listen to URL periodically
     PIDL::serveObjects();

     PIDL::finalize();
    
  }
  catch(const sci::cca::CCAException::pointer &pe) {
    std::cerr << "Caught exception:\n";
    std::cerr << pe->getNote() << std::endl;
    abort();
  }
  catch(const Exception& e) {
    std::cerr << "Caught exception:\n";
    std::cerr << e.message() << std::endl;
    abort();
  }
  catch(...) {
    std::cerr << "Caught unexpected exception!\n";
    abort();
  }
  return 0;
}


void
usage()
{
  std::cout << "Usage: plume [args] \n";
  std::cout << "      [-v | --version]                         : prints out version information\n";
  std::cout << "      [-h | --help                             : prints usage information\n";
  std::cout << "      [[-a | --attach] existing-framework-url  : attach to existing framework\n";
  std::cout << "      [[-p | --parent] parent-framework-url    : parent framework\n";
  std::cout << "      [--app] app-class-name                   : start an application\n";
  exit( 0 );
}

void add_args( const CoreFramework::pointer &framework, int argc, char *argv[])
{
  SSIDL::array1<std::string> args;
  for (int i=0; i<argc; i++)
    args.push_back(argv[i]);
  
  Properties::pointer properties = new PropertiesImpl;
  properties->putStringArray("args", args);

  Services::pointer services = framework->getServices("default", "cca.unknown", 0);
  
  services->registerUsesPort("properties", "cca.FrameworkProperties", 0);
  {
    FrameworkProperties::pointer frameworkProperties = pidl_cast<FrameworkProperties::pointer>( services->getPort("properties"));
    frameworkProperties->addProperties(properties);
  }
  services->releasePort("properties");
  services->unregisterUsesPort("properties");

  framework->releaseServices(services);
}

void
parse_args( int argc, char *argv[])
{
  for( int cnt = 0; cnt < argc; cnt++ ) {
    std::string arg( argv[ cnt ] );
    if( ( arg == "--version" ) || ( arg == "-version" )
        || ( arg == "-v" ) || ( arg == "--v" ) ) {
      std::cout << "Version: " << VERSION << std::endl;
      exit( 0 );
    } else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
		( arg == "-h" ) ||  ( arg == "--h" ) ) {
      usage();
    } 
    else if ( arg == "--parent" || arg == "-p" ) {
      if ( ++cnt == argc ) {
	std::cerr << "missing url param for " << arg << "\n";
	exit(0);
      }
      parent_url = argv[cnt];
    }
    else if ( arg == "--attach" || arg == "-a" ) {
      if ( ++cnt == argc ) {
	std::cerr << "missing url param for " << arg << "\n";
	exit(0);
      }
      attach_url = argv[cnt];
    }
    else if ( arg == "--app" ) {
      if ( ++cnt == argc ) {
	std::cerr << "missing application class name\n";
	exit(0);
      }
      app_class = argv[cnt];
    }
    else if ( arg == "--server" ) {
      server = true;
    }
  }
}

void init()
{
  create_sci_environment(0,0);
  
  PIDL::initialize();
  PIDL::isfrwk = true;
  //all threads in the framework share the same invocation id
  PRMI::setInvID(ProxyID(1,0));
}
