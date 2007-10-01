/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2007 Scientific Computing and Imaging Institute,
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
 *  main.cc: Babelized version of SCIJump
 *
 *  Written by:
 *   Kosta Damevski
 *   SCI Institute
 *   University of Utah
 *   August 2007
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Environment.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Thread/Thread.h>

#include "scijump.hxx"
#include "gov_cca.hxx"
#include "sidl.hxx"
#include "sidlx.hxx"

#include <sci_defs/mpi_defs.h>
#include <sci_mpi.h>
#include <sci_wx.h>
#include <Framework/StandAlone/scijump_version.h>
#include <iostream>

using namespace scijump;
using namespace SCIRun;
using namespace sci::cca;

#include <sys/stat.h>

static std::string defaultBuilder("gui");
static std::string fileName;

void
usage()
{
  std::cout << "Usage: sr [args] [network file]\n";
  std::cout << "       [-]-v[ersion]          : prints out version information\n";
  std::cout << "       [-]-h[elp]             : prints usage information\n";
  std::cout << "       [-]-b[uilder] gui/txt  : selects GUI or Textual builder\n";
  std::cout << "       network file           : SCIJump Network Input File\n";
  exit( 0 );
}

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???
bool
parse_args( int argc, char *argv[])
{
  bool load = false;
  for( int cnt = 0; cnt < argc; cnt++ ) {
    std::string arg( argv[ cnt ] );
    if( ( arg == "--version" ) || ( arg == "-version" )
        || ( arg == "-v" ) || ( arg == "--v" ) ) {
      std::cout << "Version: " << SCIJUMP_VERSION << std::endl;
      exit( 0 );
    } else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
                ( arg == "-h" ) ||  ( arg == "--h" ) ) {
      usage();
    } else if ( ( arg == "--builder" ) || ( arg == "-builder" ) ||
                ( arg == "-b" ) ||  ( arg == "--b" ) ) {
      if(++cnt < argc) {
        defaultBuilder = argv[cnt];
      } else {
        std::cerr << "Unknown builder."<<std::endl;
        usage();
      }
    } else {
      struct stat buf;
      if (stat(arg.c_str(),&buf) < 0) {
        std::cerr << "Couldn't find network file " << arg
                  << ".\nNo such file or directory.  Exiting." << std::endl;
        exit(0);
      } else {
        // SCIRun has new network file format -> check format and change extension
        if (ends_with(arg, ".net")) {
          fileName = arg;
          load = true;
        }
      }
    }
  }
  return load;
}

void
component_test(scijump::BuilderService& builder, ::gov::cca::Services& svc)
{
  // test framework's unique name function
  gov::cca::ComponentID helloServer = builder.createInstance("", "HelloServer.Component", NULL);
  if (helloServer._is_nil()) {
    std::cerr << "Cannot create component: babel:HelloServer\n";
    return;
  }
  gov::cca::ComponentID hello = builder.createInstance("HelloClient", "HelloClient.Component", NULL);
  if (hello._is_nil()) {
    std::cerr << "Cannot create component: babel:Hello\n";
    return;
  }

  ::gov::cca::ConnectionID msgConnID = builder.connect(hello, "msgport-up", helloServer, "msgport-pp");

  svc.registerUsesPort("goport-up", "gov.cca.ports.GoPort", 0);
  ::gov::cca::ConnectionID goConnID = builder.connect(svc.getComponentID(), "goport-up", hello, "goport");
  ::gov::cca::Port port = svc.getPort("goport-up");
  ::gov::cca::ports::GoPort goPort = ::sidl::babel_cast< ::gov::cca::ports::GoPort>(port);
  goPort.go();

  builder.disconnect(goConnID, 0);
  builder.disconnect(msgConnID, 0);

  svc.releasePort("goport-up");
  svc.unregisterUsesPort("goport-up");

  builder.destroyInstance(hello, 0);
  builder.destroyInstance(helloServer, 0);
}

void
local_framework_test(scijump::BuilderService& builder, ::gov::cca::Services& svc)
{
  gov::cca::ComponentID dvr = builder.createInstance("DriverInstance", "drivers.CXXDriver", 0);
  if(dvr._is_nil()) {
    std::cerr << "Cannot create component: babel:DriverInstance\n";
    return;
  }

  gov::cca::ComponentID intg = builder.createInstance("IntegratorInstance", "integrators.MonteCarlo", 0);
  if(intg._is_nil()) {
    std::cerr << "Cannot create component: babel:IntegratorInstance\n";
    return;
  }

  gov::cca::ComponentID func = builder.createInstance("FunctionInstance", "functions.PiFunction", 0);
  if(func._is_nil()) {
    std::cerr << "Cannot create component: babel:FunctionInstance\n";
    return;
  }

  gov::cca::ComponentID rand = builder.createInstance("RandomNumGeneratorInstance", "randomgens.RandNumGenerator", 0);
  if(rand._is_nil()) {
    std::cerr << "Cannot create component: babel:RandonNumGeneratorInstance\n";
    return;
  }

  ::gov::cca::ConnectionID intgConnID = builder.connect(dvr,"IntegratorPort-up",intg,"IntegratorPort-pp");
  ::gov::cca::ConnectionID funcConnID = builder.connect(intg,"FunctionPort-up",func,"FunctionPort-pp");
  ::gov::cca::ConnectionID randConnID = builder.connect(intg,"RandomGeneratorPort-up",rand,"RandomGeneratorPort-pp");


  svc.registerUsesPort("goport-up", "gov.cca.ports.GoPort", 0);
  ::gov::cca::ConnectionID goConnID = builder.connect(svc.getComponentID(), "goport-up", dvr, "GoPort");
  ::gov::cca::Port port = svc.getPort("goport-up");
  ::gov::cca::ports::GoPort goPort = ::sidl::babel_cast< ::gov::cca::ports::GoPort>(port);
  if(goPort._not_nil()) {
    goPort.go();
  }

  builder.disconnect(goConnID, 0);
  builder.disconnect(intgConnID, 0);
  builder.disconnect(funcConnID, 0);
  builder.disconnect(randConnID, 0);

  svc.releasePort("goport-up");
  svc.unregisterUsesPort("goport-up");

  builder.destroyInstance(dvr, 0);
  builder.destroyInstance(intg, 0);
  builder.destroyInstance(func, 0);
  builder.destroyInstance(rand, 0);
}


int
orbStart(sidlx::rmi::SimpleOrb& echo, int port_number)
{
  sidl::rmi::ProtocolFactory pf;
  if(!pf.addProtocol("simhandle","sidlx.rmi.SimHandle")) {
    std::cout << "Error in addProtocol\n";
    exit(2);
  }

  echo.requestPort(port_number);
  int tid = echo.run();
  sidl::rmi::ServerRegistry::registerServer(echo);
  return tid;
}


int
main(int argc, char *argv[], char **environment) {
  bool startFramework = true;
  bool loadNet = parse_args(argc, argv);
  
  // Set up a default environment for Babel.
  // This should be moved to a framework properties service.
  create_sci_environment(environment, 0);
  std::string o(sci_getenv("SCIRUN_OBJDIR"));
  ASSERT(! o.empty());
  std::string p = o + "/Components/Babel/xml/BabelComponents.scl";
  ::sidl::Loader::addSearchPath(p);

  int orb_port_num = 22222;

  // Create a new framework
  try {
    sidlx::rmi::SimpleOrb echo = sidlx::rmi::SimpleOrb::_create();
    int tid = orbStart(echo,orb_port_num);
    SCIJumpFramework sj;

    if (startFramework) {
      sj = SCIJumpFramework::_create();

      std::cerr << "URL to framework:\n" << sj._getURL() << std::endl;
      //std::ofstream f("framework.url");
      //std::string s;
      //f << sr->getURL().getString();
      //f.close();
    } else {
      std::cerr << "Not finished: pass url to existing framework" << std::endl;
    }

    gov::cca::TypeMap mainProperties = sj.createTypeMap();
    // TODO: Is this property still needed?
    mainProperties.putBool("internal component", true);
    gov::cca::Services mainServices = sj.getServices("SCIJump main", "main", mainProperties);
    mainServices.registerUsesPort("mainBuilder", "cca.BuilderService", mainServices.createTypeMap());

    ::gov::cca::Port bsp = mainServices.getPort("mainBuilder");
    ASSERT(bsp._not_nil());
    scijump::BuilderService builder = babel_cast<scijump::BuilderService>(bsp);
    ASSERT(builder._not_nil());

    /*
    gov::cca::ports::FrameworkProperties fwkProperties = mainServices.getPort("cca.FrameworkProperties");
    if(fwkProperties._is_nil()) {
      std::cerr << "Fatal Error: Cannot find framework properties service\n";
      exit(1);
    }


    if (loadNet) {
      TypeMap map = fwkProperties.getProperties();
      map.putString("network file", fileName);
      fwkProperties.setProperties(map);
    }
    */

//#if !defined(HAVE_WX)
  //defaultBuilder = "txt";
//#endif

    /*
    if (defaultBuilder == "gui") {
      TypeMap guiProperties = sj.createTypeMap();
      guiProperties.putBool("internal component", true);
      ComponentID gui_id = builder.createInstance("SCIRun.GUIBuilder",
                                                  "cca:SCIRun.GUIBuilder",
                                                  guiProperties);
      if (gui_id._is_nil()) {
        std::cerr << "Cannot create component: cca:SCIRun.GUIBuilder\n";
        Thread::exitAll(1);
      }
    } else {

      ComponentID gui_id = builder.createInstance("TxtBuilder", "cca:SCIRun.TxtBuilder",0);
      if(gui_id._is_nil()) {
        std::cerr << "Cannot create component: cca:SCIRun.TxtBuilder\n";
        Thread::exitAll(1);
      }

    }
    mainServices.releasePort("cca.FrameworkProperties");
    mainServices.releasePort("cca.BuilderService");
    */

    std::cout << "\nSCIJump " << SCIJUMP_VERSION << " started..." << std::endl;

    // test instantiation and connection
    component_test(builder, mainServices);
    local_framework_test(builder, mainServices);
    mainServices.releasePort("mainBuilder");

    //broadcast, listen to URL periodically
    //sr->share(mainServices);

    // test, although should be in cleanup code
    sj.releaseServices(mainServices);
    sj.shutdownFramework();
    // test, although should be in cleanup code
  }
  catch (sidl::RuntimeException& e) {
    std::cerr << "Caught a SIDL runtime exception with note: " << e.getNote() << std::endl;
    std::cerr << e.getTrace() << std::endl;
    abort();
  }
  catch (gov::cca::CCAException& e) {
    std::cerr << "Caught a CCA exception with note: " << e.getNote() << std::endl;
    std::cerr << e.getTrace() << std::endl;
    abort();
  }
  catch (...) {
    std::cerr << "Caught unexpected exception!\n";
    abort();
  }
  return 0;
}
