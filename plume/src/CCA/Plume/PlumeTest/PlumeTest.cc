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
 *  PlumeTest.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   October 2005
 *
 */
#include <Core/CCA/spec/sci_sidl.h>
#include <CCA/Plume/PlumeTest/PlumeTest.h>
#include <SCIRun/Core/CCAException.h>

extern "C" sci::cca::Component::pointer make_cca_plume_PlumeTest()
{
    return sci::cca::Component::pointer(new SCIRun::PlumeTest());
}

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::ports;

  PlumeTest::PlumeTest() {}

  PlumeTest::~PlumeTest() 
  {
    if ( !services.isNull() ) {
      if ( !goPort.isNull() ) services->unregisterUsesPort("go");
      //framework->releaseServices(services);
    }
  }

  void PlumeTest::setServices(const Services::pointer& svc)
  {
    services = svc;
    goPort = new PlumeTestPort(this);
    services->addProvidesPort(goPort, "go", "sci.cca.ports.GoPort", 0);
  }

  int PlumeTest::go()
  {
    try {
      std::cerr << "PlumeTest: setup\n";
      services->registerUsesPort("runTest", "sci.cca.ports.GoPort", 0);  
      services->registerUsesPort("builder", "cca.BuilderService", 0);

      std::cout << "PlumeTest: get builder.\n";
      BuilderService::pointer builder = 
	pidl_cast<BuilderService::pointer>( services->getPort("builder"));
    
      std::cout << "PlumeTest: create 'Hello' component\n";
      ComponentID::pointer hello = builder->createInstance("hello", "cca.core.Hello", 0);

      std::cout << "PlumeTest: create 'World' component\n";
      ComponentID::pointer world  = builder->createInstance("world", "cca.core.World", 0);
    
      std::cout << "PlumeTest: connect components\n";
      ConnectionID::pointer helloworldConnection =
	builder->connect( hello, "hello", world, "message");
    
      std::cout << "PlumeTest: connet to Hello::go\n";
      ConnectionID::pointer goConnection = 
	builder->connect(services->getComponentID(), "runTest", hello, "go");
      GoPort::pointer test = pidl_cast<GoPort::pointer>(services->getPort("runTest"));
    
      std::cout << "PlumeTest: run\n\n";
      test->go();
    
      std::cout << "\nPlumeTest: cleanup\n";
      builder->disconnect( goConnection, 0 );

      std::cout << "PlumeTest: remove components\n";
      builder->disconnect(helloworldConnection, 0);
      builder->destroyInstance(hello, 0);
      builder->destroyInstance(world, 0);
      
      std::cout << "PlumeTest: disconnect from Frameowrk\n";
      services->releasePort("builder");
      services->unregisterUsesPort("builder");

      std::cout << "PlumeTest: done\n";
    }
    catch ( const SCIRun::CCAException::pointer &e ) {
      std::cerr << "PlumeTest: caught CCAExcetion: " << e->getNote() << "\n"
		<< "Backtrace: \n";
      dynamic_cast<SCIRun::CCAException *>(e.getPointer())->show_stack();
      return -1;
    }

    return 0;
  }

} // namespace 
