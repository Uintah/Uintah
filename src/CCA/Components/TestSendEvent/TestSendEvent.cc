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


#include <CCA/Components/TestSendEvent/TestSendEvent.h>
#include <SCIRun/Internal/Topic.h>
#include <SCIRun/Internal/Event.h>
#include <SCIRun/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestSendEvent()
{
  return sci::cca::Component::pointer(new TestSendEvent());
}


TestSendEvent::TestSendEvent()
{
}

TestSendEvent::~TestSendEvent()
{
  services->removeProvidesPort("go");
}

void TestSendEvent::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestSendEventgo *providesgp = new TestSendEventgo();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestSendEventgo::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestSendEventgo::go()
{
  std::cout<<"Inside Go Function of TestSendEvent\n";
  sci::cca::ports::EventService::pointer ptr;
  try {
    sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
    ptr = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
    if (ptr.isNull())
      std::cout << "Pointer returned from getPort is Null!!!\n";
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cout << e->getNote() << std::endl;
    return 1;
  }
  sci::cca::TypeMap::pointer typeMapPtr;
  //Send Event with an empty event body
  std::cout << "Test 1 : Send an Event with an empty Header\n";
  try {
    sci::cca::Topic::pointer topicPtr = ptr->createTopic("Hello");
    sci::cca::TypeMap::pointer header = com->getServices()->createTypeMap();
    sci::cca::TypeMap::pointer body = com->getServices()->createTypeMap();
    sci::cca::Event::pointer eventPtr(new Event(header, body));
    std::cout << "Before send event call\n";
    topicPtr->sendEvent(eventPtr);
    std::cout << "After send Event call\n";
  } catch(const sci::cca::EventServiceException::pointer &e) {
    std::cout << "Exception in trying to send an Event with empty Header" << e->getNote() << std::endl;
  }

  //Send an Event successfully
  std::cout << "Test 2: Send an Event successfully\n";
  try {
    sci::cca::Topic::pointer topicPtr = ptr->createTopic("Hello");
    sci::cca::TypeMap::pointer header = com->getServices()->createTypeMap();
    sci::cca::TypeMap::pointer body = com->getServices()->createTypeMap();
    sci::cca::Event::pointer eventPtr(new Event(header, body));
    topicPtr->sendEvent(eventPtr);
    std::cout << "Send Event successful\n";
  } catch(const sci::cca::EventServiceException::pointer &e) {
    std::cout << e->getNote() << std::endl;
  }
  return 0;
}

