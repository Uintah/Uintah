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
 *  Plume.cc: Dugway project
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   June 2005
 *
 */
#include <iostream>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunFramework.h>

#include <Packages/Plume/StandAlone/Config.h>
#include <Packages/Plume/StandAlone/Plume.defs>

using namespace SCIRun;
using namespace sci::cca;

#include <sys/stat.h>

int
main(int argc, char *argv[]) 
{
  if ( !init(argc,argv) ) return 0;

  Dugway::Config& config = Dugway::ProgramOptions::Instance();

  // Create a new framework
  try {
    AbstractFramework::pointer sr;
    if(config.framework == "") {
      sr = AbstractFramework::pointer(new SCIRunFramework());
      std::cerr << "URL to framework:\n" << sr->getURL().getString() << std::endl;
    } else {
      std::cerr << "Not finished: connect to existing framework\n";
      return 0;
    }
    
    sci::cca::Services::pointer main_services
      = sr->getServices("SCIRun main", "main", sci::cca::TypeMap::pointer(0));

    sci::cca::ports::FrameworkProperties::pointer properties =
      pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(
			main_services->getPort("cca.FrameworkProperties"));
    if (properties.isNull()) {
      std::cerr << "Cannot find framework properties service\n";
      Thread::exitAll(1);
    }

    sci::cca::ports::BuilderService::pointer builder_service
      = pidl_cast<sci::cca::ports::BuilderService::pointer>(
                          main_services->getPort("cca.BuilderService"));
    if(builder_service.isNull()) {
      std::cerr << "Cannot find builder service\n";
      Thread::exitAll(1);
    }
    
    ComponentID::pointer builder =
      builder->createInstance(config.builder.first, config.builder.second, sci::cca::TypeMap::pointer(0));
    if(builder.isNull()) {
      std::cerr << "Cannot create builder " << config.builder.first << " of type " << config.builder.second << std::endl;
      Thread::exitAll(1);
    }

    main_services->releasePort("cca.FrameworkProperties");
    main_services->releasePort("cca.BuilderService");
    std::cout << "Plume ready" << std::endl;
    
    //broadcast, listen to URL periodically
    //sr->share(main_services);
    
    PIDL::serveObjects();
    std::cout << "serveObjects done!\n";
    PIDL::finalize();
    
  }
  catch(const Exception& e) {
    std::cerr << "Plume: " <<  e.message() << std::endl;
    return 1;
  }

  return 0;
}

bool init(int argc, char *argv[])
{
  // parse command line,and config file options
  if ( !Dugway::ProgramOptions::Instance().init( argc, argv ) ) return false;

  // PIDL
  try {
    PIDL::initialize();
    PIDL::isfrwk=true;
    //all threads in the framework share the same
    //invocation id
    PRMI::setInvID(ProxyID(1,0));
  }
  catch(const Exception& e) {
    std::cerr << "PIDL: " << e.message() << std::endl;
    return false;
  }
}  
