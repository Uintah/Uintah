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


#include<CCA/Components/TestRegisterEventListener/TestRegisterEventListener.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestRegisterEventListener()
{
  return sci::cca::Component::pointer(new TestRegisterEventListener());
}

void TestEventListener::processEvent(const std::string &topicName, const sci::cca::Event::pointer &theEvent)
{
  //std::cout << eventBody->getString("Event1",std::string("default"));
  sci::cca::TypeMap::pointer eventBody = theEvent->getBody();
  SSIDL::array1<std::string> allKeys = eventBody->getAllKeys(sci::cca::None);
  for (unsigned int i = 0; i < allKeys.size(); i++)
      std::cout << "sampcomp1: Topic Name: " << topicName << "\tEvent Body: " << eventBody->getString(allKeys[i],std::string("default")) << std::endl;
}
TestRegisterEventListener::TestRegisterEventListener()
{
}

TestRegisterEventListener::~TestRegisterEventListener()
{
  services->removeProvidesPort("go");
}

void TestRegisterEventListener::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestRegisterEventListenergo *providesgp = new TestRegisterEventListenergo();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestRegisterEventListenergo::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestRegisterEventListenergo::go()
{
  std::cout<<"Inside Go Function of TestRegisterEventListener\n";
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
   sci::cca::EventListener::pointer evptr(new TestEventListener);
   sci::cca::Topic::pointer topicPtr = ptr->createTopic("Hello");
   //Register event Listener with empty listener key
   std::cout << "Test 1 : Register event Listener with empty listener key:\n";
   try
     {
       topicPtr->registerEventListener(std::string(""),evptr);
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in trying to register event listener with empty listener key: " << e->getNote() << std::endl;
     }
   
   //Register event listener with an already existing listener key
   std::cout << "Test 2 : Register event listener with an already existing listener key\n";
   try
     {
       topicPtr->registerEventListener(std::string("sample"),evptr);
       topicPtr->registerEventListener(std::string("sample"),evptr);
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in trying to register event listener with an already existing listener key" << e->getNote() << std::endl;
     }
   //Register Event Listener with a null eventlistener pointer
   std::cout << "Test 3: Register Event Listener with a null eventlistener pointer\n";
   try
     {
       sci::cca::EventListener::pointer samplePtr;
       topicPtr->registerEventListener(std::string("Test"),samplePtr);
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in registering event listener with null pointer " << e->getNote() << std::endl;
     }
   //Register an Event Listener
   std::cout << "Test 4: Register an Event Listener\n";
   try
     {
       topicPtr->registerEventListener(std::string("Test"),evptr);
       std::cout << "Registered successfully\n";
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in registering an event listener  " << e->getNote() << std::endl;
     }
   return 0;
}

