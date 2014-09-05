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
 *  Lesson03.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   October 2005
 *
 */
#include <Core/CCA/spec/sci_sidl.h>
#include <CCA/TENA/Lesson03/Lesson03.h>
#include <SCIRun/Core/CCAException.h>

extern "C" sci::cca::Component::pointer make_cca_tena_lesson03_Lesson03()
{
    return sci::cca::Component::pointer(new SCIRun::Lesson03());
}

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::ports;

  Lesson03::Lesson03() {}

  Lesson03::~Lesson03() 
  {
    if ( !services.isNull() ) {
      if ( !goPort.isNull() ) services->unregisterUsesPort("go");
      //framework->releaseServices(services);
    }
  }

  void Lesson03::setServices(const Services::pointer& svc)
  {
    services = svc;
    goPort = new Lesson03Port(this);
    services->addProvidesPort(goPort, "go", "sci.cca.ports.GoPort", 0);
  }

  int Lesson03::go()
  {
    try {
      std::cerr << "Lesson03: setup\n";
      services->registerUsesPort("runTest", "sci.cca.ports.GoPort", 0);  
      services->registerUsesPort("builder", "cca.BuilderService", 0);

      std::cout << "Lesson03: get builder.\n";
      BuilderService::pointer builder = 
	pidl_cast<BuilderService::pointer>( services->getPort("builder"));
    
      std::cout << "Lesson03: create 'SubscribePerson' component\n";
      ComponentID::pointer subscribe = builder->createInstance("subscribe", "cca.tena.lesson03.SubscribePerson", 0);

      std::cout << "Lesson03: create 'PublishPerson' component\n";
      ComponentID::pointer publish = builder->createInstance("publish", "cca.tena.lesson03.PublishPerson", 0);
    
      std::cout << "Lesson03: connet to Publish::go\n";
      ConnectionID::pointer goConnection = 
	builder->connect(services->getComponentID(), "runTest", publish, "go");
      GoPort::pointer test = pidl_cast<GoPort::pointer>(services->getPort("runTest"));
    
      std::cout << "Lesson03: run\n\n";
      test->go();
    
      std::cout << "\nLesson03: cleanup\n";
      builder->disconnect( goConnection, 0 );

      std::cout << "Lesson03: remove components\n";
      builder->destroyInstance(publish, 0);
      builder->destroyInstance(subscribe, 0);
      
      std::cout << "Lesson03: disconnect from Frameowrk\n";
      services->releasePort("builder");
      services->unregisterUsesPort("builder");

      std::cout << "Lesson03: done\n";
    }
    catch ( const SCIRun::CCAException::pointer &e ) {
      std::cerr << "Lesson03: caught CCAExcetion: " << e->getNote() << "\n"
		<< "Backtrace: \n";
      dynamic_cast<SCIRun::CCAException *>(e.getPointer())->show_stack();
      return -1;
    }

    return 0;
  }

} // namespace 
