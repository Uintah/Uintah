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


#include<CCA/Components/TestSubscription/TestSubscription.h>
#include<Framework/Internal/Topic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestSubscription()
{
  return sci::cca::Component::pointer(new TestSubscription());
}
void TestSubscriptionEventListener::processEvent(const std::string &topicName, const sci::cca::Event::pointer &theEvent)
{
  try{
    sci::cca::TypeMap::pointer eventBody = theEvent->getBody();
     SSIDL::array1<std::string> allKeys = eventBody->getAllKeys(sci::cca::None);
     for (unsigned int i = 0; i < allKeys.size(); i++)
        std::cout << "TestWildcard: Topic Name: " << topicName << "\tEvent Body: " << eventBody->getString(allKeys[i],std::string("default")) << std::endl; 
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exception in Processing Events" << e->getNote() << std::endl;
  }
}

TestSubscription::TestSubscription()
{
}

TestSubscription::~TestSubscription()
{
  services->removeProvidesPort("go");
}

void TestSubscription::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestSubscriptionGo *providesgp0 = new TestSubscriptionGo();
  providesgp0->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestSubscriptionGo::pointer(providesgp0), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestSubscriptionGo::go()
{
  std::cout<<"Inside Go Function of TestSubscription\n";
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
  sci::cca::EventListener::pointer evptr(new TestSubscriptionEventListener);

  sci::cca::Topic::pointer topicPtr = ptr->createTopic("randomTopic");
  sci::cca::Topic::pointer topicPtr1 = ptr->createTopic("test.Echo.one");
  sci::cca::Topic::pointer topicPtr2 = ptr->createTopic("test.Echo.two");
  sci::cca::Topic::pointer topicPtr3 = ptr->createTopic("test.random.one");

  //  Register a Listener to a Subscription (with a '*' at the beginning)
  std::cout << "Test 1 : Register a Listener to a Subscription(with a '*' at the beginning)\n";
  try{
    sci::cca::Subscription::pointer subPtr1(ptr->subscribeToEvents("*.one"));
    std::cout << "Subscription successfully created\n";
    subPtr1->registerEventListener(std::string("SubscriptionListener"),evptr);
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a Subscription : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }
   

  //  Register a Listener to a Subscription (which has a '%')
  std::cout << "Test 2 : Register a Listener to a Subscription(with a '*' in the middle)\n";
  try{
    sci::cca::Subscription::pointer subPtr2(ptr->subscribeToEvents("test.%.one"));
    subPtr2->registerEventListener(std::string("SubscriptionListener1"),evptr);
    
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a Subscription : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }
  

  //  Register a Listener to a Subscription (with a '*' at the end)
  std::cout << "Test 3 : Register a Listener to a Subscription(with a '*' at the end)\n";
  try{
    sci::cca::Subscription::pointer subPtr3(ptr->subscribeToEvents("test.*"));
    subPtr3->registerEventListener(std::string("SubscriptionListener2"),evptr);
    
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a Subscription : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }


  //Send Events
  try {
     sci::cca::TypeMap::pointer eventBody = com->getServices()->createTypeMap();
     eventBody->putString(std::string("Event1"),std::string("Event sent to randomTopic"));
     topicPtr->sendEvent("randomTopic",eventBody);

     sci::cca::TypeMap::pointer eventBody1 = com->getServices()->createTypeMap();
     eventBody1->putString(std::string("Event1"),std::string("Event sent to test.Echo.one"));
     topicPtr1->sendEvent("test.Echo.one",eventBody1);

     sci::cca::TypeMap::pointer eventBody2 = com->getServices()->createTypeMap();
     eventBody2->putString(std::string("Event2"),std::string("Event sent to test.Echo.two"));
     topicPtr2->sendEvent("test.Echo.two",eventBody2);

     sci::cca::TypeMap::pointer eventBody3 = com->getServices()->createTypeMap();
     eventBody3->putString(std::string("Event3"),std::string("Event sent to test.random.one"));
     topicPtr3->sendEvent("test.random.one",eventBody3);
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "EventService Exception : " << e->getNote() << std::endl;
  }
  return 0;
}

