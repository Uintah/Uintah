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


#include<CCA/Components/TestEchoEvent/TestEchoEvent.h>
#include<Framework/Internal/Topic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestEchoEvent()
{
  return sci::cca::Component::pointer(new TestEchoEvent());
}

void TestEchoEventListener::processEvent(const std::string &topicName, const sci::cca::Event::pointer &theEvent)
{
  try{
    //std::cout << eventBody->getString("Event1",std::string("default"));
    sci::cca::TypeMap::pointer eventBody = theEvent->getBody();
    SSIDL::array1<std::string> allKeys = eventBody->getAllKeys(sci::cca::None);
    for (unsigned int i = 0; i < allKeys.size(); i++)
      std::cout << "TestEchoEvent: Topic Name: " << topicName << "\tEvent Body: " << eventBody->getString(allKeys[i],std::string("default")) << std::endl; 
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "EventServiceException : "<<e->getNote() << std::endl;
  }
}

TestEchoEvent::TestEchoEvent()
{
}

TestEchoEvent::~TestEchoEvent()
{
  services->removeProvidesPort("go");
}

void TestEchoEvent::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestEchoEventgo *providesgp = new TestEchoEventgo();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestEchoEventgo::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestEchoEventgo::go()
{
  std::cout<<"Inside Go Function of TestEchoEvent\n";
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
  try{
    sci::cca::EventListener::pointer echoEvptr(new TestEchoEventListener);
    std::string sendString("Sample Text");
    //Test SendEvent
    std::cout << "Test 1 : Test SendEvent\n"; 
    sci::cca::Topic::pointer topicPtr = ptr->createTopic("SampleTopic");
    std::cout << "Topic created successfully.\n";

    //Test Register EventListener
    std::cout << "Test 2: Test Register EventListener\n";
    sci::cca::Subscription::pointer subPtr = ptr->subscribeToEvents("SampleTopic");
    subPtr->registerEventListener(std::string("EchoEventListener"),echoEvptr);
    std::cout << "Listener registered successfully.\n";
    
    //Test SendEvent
    std::cout << "Test 3 : Test SendEvent\n";
    sci::cca::TypeMap::pointer eventBody = com->getServices()->createTypeMap();
    eventBody->putString(std::string("Event1"),sendString);
    topicPtr->sendEvent("SampleTopic",eventBody);
    std::cout << "Event sent successfully.\n" << std::endl;
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "EventService Exception: " << e->getNote() << std::endl;
  }
  return 0;
}

