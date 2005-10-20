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
 *  Core.cc: CCA-ified version of SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Util/Environment.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Thread/Thread.h>

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Core/TypeMapImpl.h>
#include <SCIRun/Core/CoreFrameworkImpl.h>

#include <CCA/Core/Hello/Hello.h>
#include <SCIRun/Core/SimpleComponentClassFactory.h>

#include <sci_defs/qt_defs.h>
#include <iostream>

using namespace SCIRun;
using namespace sci::cca;
using namespace sci::cca::ports;

#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml
#include <sys/stat.h>

std::string defaultBuilder = "gui";
static std::string fileName;

void init();
void usage();
bool parse_args( int argc, char *argv[]);

int
main(int argc, char *argv[]) {
  bool framework = true;
  
  init();
  
  // Create a new framework
  try {
    CoreFramework::pointer core;
    if(framework) {
      core = CoreFramework::pointer(new CoreFrameworkImpl());
      //std::cerr << "URL to framework:\n" << core->getURL().getString() << std::endl;
    } else {
      std::cerr << "Not finished: pass url to existing framework\n";
    }

    ComponentClassFactory::pointer hello( new SimpleComponentClassFactory<Hello>("core.example.hello") ); 
    core->addComponentClassFactory( hello );
	     
    Services::pointer main_services = core->getServices("main", "cca.unknown", core->createTypeMap());
    main_services->registerUsesPort("builder", "cca.BuilderService", core->createTypeMap());
    
    BuilderService::pointer builder = pidl_cast<BuilderService::pointer>( main_services->getPort("builder"));

    main_services->releasePort("builder");

    main_services->unregisterUsesPort("builder");
    std::cout << "Core " << VERSION << " started..." << std::endl;
  
    //broadcast, listen to URL periodically
    
    PIDL::serveObjects();
    std::cout << "serveObjects done!\n";
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
  std::cout << "Usage: plume [args] [network file]\n";
  std::cout << "       [-]-v[ersion]          : prints out version information\n";
  std::cout << "       [-]-h[elp]             : prints usage information\n";
  std::cout << "       [-]-b[uilder] gui/txt  : selects GUI or Textual builder\n";
  std::cout << "       network file           : SCIRun Network Input File\n";
  exit( 0 );
}

bool
parse_args( int argc, char *argv[])
{
  bool ok = false;
  for( int cnt = 0; cnt < argc; cnt++ ) {
    std::string arg( argv[ cnt ] );
    if( ( arg == "--version" ) || ( arg == "-version" )
        || ( arg == "-v" ) || ( arg == "--v" ) ) {
      std::cout << "Version: " << VERSION << std::endl;
      exit( 0 );
    } else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
              ( arg == "-h" ) ||  ( arg == "--h" ) ) {
      usage();
    } else if ( ( arg == "--builder" ) || ( arg == "-builder" ) ||
              ( arg == "-b" ) ||  ( arg == "--b" ) ) {
      if(++cnt<argc) {
        defaultBuilder=argv[cnt];
      } else {
        std::cerr << "Unkown builder."<<std::endl;
        usage();
      }
    } else {
      struct stat buf;
      if (stat(arg.c_str(),&buf) < 0) {
        std::cerr << "Couldn't find network file " << arg
                  << ".\nNo such file or directory.  Exiting." << std::endl;
        exit(0);
      } else {
          if (ends_with(arg, ".net")) {
              fileName = arg;
              ok = true;
          }
      }
    }
  }
  return ok;
}

void init()
{
  create_sci_environment(0,0);
  
  try {
    // TODO: Move this out of here???
    PIDL::initialize();
    PIDL::isfrwk = true;
    //all threads in the framework share the same
    //invocation id
    PRMI::setInvID(ProxyID(1,0));
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
}
