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


#include<CCA/Components/TestUnregisterEventListener/TestUnregisterEventListener.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestUnregisterEventListener()
{
  return sci::cca::Component::pointer(new TestUnregisterEventListener());
}


TestUnregisterEventListener::TestUnregisterEventListener()
{
}

TestUnregisterEventListener::~TestUnregisterEventListener()
{
  services->removeProvidesPort("go");
}

void TestUnregisterEventListener::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestUnregisterEventListenergo *providesgp = new TestUnregisterEventListenergo();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestUnregisterEventListenergo::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestUnregisterEventListenergo::go()
{
   std::cout<<"Inside Go Function of TestUnregisterEventListener\n";
   sci::cca::ports::EventService::pointer ptr;
   try {
      sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
      ptr = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
      if(ptr.isNull())
	std::cout << "Pointer is Null!!!\n";
   }
    catch (const sci::cca::CCAException::pointer &e) {
      std::cout << e->getNote() << std::endl;
    return 1;
  }
   sci::cca::Topic::pointer topicPtr = ptr->createTopic("Hello");
   //Unregister with empty topic name
   std::cout << "Test 1 : Unregister with empty topic name\n";
   try
     {
       topicPtr->unregisterEventListener(std::string(""));
     }
   catch(const sci::cca::EventServiceException::pointer  &e)
     {
       std::cout << "Exception in trying to unregister with empty topic name: " << e->getNote() << std::endl;
     }
   //Unregister with non-existent topic namew
   std::cout << "Test 2 : Unregister a non-existent topic name\n";
   try
     {
       topicPtr->unregisterEventListener("Random");
  
     }
   catch(const sci::cca::EventServiceException::pointer  &e)
     {
       std::cout << "Exception in trying to unregister a non-existent topic name: " << e->getNote() << std::endl;
     }
  //Unregister with non-existent topic namew
   std::cout << "Test 3 : Unregister an existent topic \n";
   try
     {
       topicPtr->unregisterEventListener("Hello");
       std::cout << "Unregister successful\n";
     }
   catch(const sci::cca::EventServiceException::pointer  &e)
     {
       std::cout << "Exception in trying to unregister an existent topic name: " << e->getNote() << std::endl;
     }
   return 0;
     
}

