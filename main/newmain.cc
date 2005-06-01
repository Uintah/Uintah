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
 *  newmain.cc: CCA-ified version of SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>
#include <sci_defs/mpi_defs.h>
#include <mpi.h>
#include <sci_defs/qt_defs.h>

using namespace SCIRun;
using namespace sci::cca;

#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml
#include <sys/stat.h>

std::string defaultBuilder = "qt";

void
usage()
{
  std::cout << "Usage: scirun [args] [net_file]\n";
  std::cout << "       [-]-v[ersion]          : prints out version information\n";
  std::cout << "       [-]-h[elp]             : prints usage information\n";
  std::cout << "       [-]-b[uilder] qt/txt   : selects QT or Textual builder\n";
  std::cout << "       net_file               : SCIRun Network Input File\n";
  exit( 0 );
}

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???
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
        std::cerr << "Couldn't find net file " << arg
                  << ".\nNo such file or directory.  Exiting." << std::endl;
        exit(0);
      }
    }
  }
}

int
main(int argc, char *argv[]) {
  bool framework = true;
  //  MPI_Init(&argc,&argv);
  
  parse_args( argc, argv);
  
  try {
    // TODO: Move this out of here???
    PIDL::initialize();
    PIDL::isfrwk=true;
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
  
  // Create a new framework
  try {
    AbstractFramework::pointer sr;
    if(framework) {
      sr = AbstractFramework::pointer(new SCIRunFramework());
      std::cerr << "URL to framework:\n" << sr->getURL().getString() << std::endl;
      //ofstream f("framework.url");
      //std::string s;
      //f<<sr->getURL().getString();
      //f.close();
    } else {
      std::cerr << "Not finished: pass url to existing framework\n";
    }
    
    sci::cca::Services::pointer main_services
      = sr->getServices("SCIRun main", "main", sci::cca::TypeMap::pointer(0));

    sci::cca::ports::FrameworkProperties::pointer fwkProperties =
      pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(
			main_services->getPort("cca.FrameworkProperties"));
    if (fwkProperties.isNull()) {
      std::cerr << "Fatal Error: Cannot find framework properties service\n";
      Thread::exitAll(1);
    }

    sci::cca::ports::BuilderService::pointer builder
      = pidl_cast<sci::cca::ports::BuilderService::pointer>(
                          main_services->getPort("cca.BuilderService"));
    if(builder.isNull()) {
      std::cerr << "Fatal Error: Cannot find builder service\n";
      Thread::exitAll(1);
    }
    
#   if !defined(HAVE_QT)
    defaultBuilder="txt";
#   endif
    
    if(defaultBuilder=="qt") {
      ComponentID::pointer gui_id =
        builder->createInstance("QtBuilder", "cca:SCIRun.Builder",
                                  sci::cca::TypeMap::pointer(0));
      if(gui_id.isNull()) {
        std::cerr << "Cannot create component: cca:SCIRun.Builder\n";
        Thread::exitAll(1);
      }
    } else {
      ComponentID::pointer gui_id =
        builder->createInstance("TxtBuilder", "cca:SCIRun.TxtBuilder",
                                sci::cca::TypeMap::pointer(0));
      if(gui_id.isNull()) {
        std::cerr << "Cannot create component: cca:SCIRun.TxtBuilder\n";
        Thread::exitAll(1);
      }
    }
    main_services->releasePort("cca.FrameworkProperties");
    main_services->releasePort("cca.BuilderService");
    std::cout << "SCIRun " << VERSION << " started..." << std::endl;
  
    //broadcast, listen to URL periodically
    //sr->share(main_services);
    
    PIDL::serveObjects();
    std::cout << "serveObjects done!\n";
    PIDL::finalize();
    
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
