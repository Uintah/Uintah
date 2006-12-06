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


#include<CCA/Components/TestWildcardTopic/TestWildcardTopic.h>
#include<Framework/Internal/Topic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestWildcardTopic()
{
  return sci::cca::Component::pointer(new TestWildcardTopic());
}
void TestWildcardTopicEventListener::processEvent(const std::string &topicName, const sci::cca::Event::pointer &theEvent)
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

TestWildcardTopic::TestWildcardTopic()
{
}

TestWildcardTopic::~TestWildcardTopic()
{
  services->removeProvidesPort("go");
}

void TestWildcardTopic::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestWildcardtopicgo *providesgp0 = new TestWildcardtopicgo();
  providesgp0->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestWildcardtopicgo::pointer(providesgp0), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestWildcardtopicgo::go()
{
  std::cout<<"Inside Go Function of TestWildcardTopic\n";
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
  sci::cca::EventListener::pointer evptr(new TestWildcardTopicEventListener);

  sci::cca::Topic::pointer topicPtr = ptr->createTopic("randomTopic");
  sci::cca::Topic::pointer topicPtr1 = ptr->createTopic("test.Echo.one");
  sci::cca::Topic::pointer topicPtr2 = ptr->createTopic("test.Echo.two");
  sci::cca::Topic::pointer topicPtr3 = ptr->createTopic("test.random.one");
  //Create a WildcardTopic without a wildcard character
  std::cout << "Test 1 : Creating a Wildcardtopic\n";
  try{
    sci::cca::WildcardTopic::pointer wildcardTopicPtr(ptr->createWildcardTopic("test.Echo.one"));
    std::cout << "WildcardTopic successfully created\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exception in creating a WildcardTopic " << e->getNote() << std::endl; 
  }

  //  Register a Listener to a Wildcardtopic (with a '*' at the beginning)
  std::cout << "Test 1 : Register a Listener to a Wildcardtopic(with a '*' at the beginning)\n";
  try{
    sci::cca::WildcardTopic::pointer wildcardTopicPtr1(ptr->createWildcardTopic("*.one"));
    std::cout << "WildcardTopic successfully created\n";
    wildcardTopicPtr1->registerEventListener(std::string("WildcardTopicListener"),evptr);
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a WildcardTopic : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }
   

  //  Register a Listener to a Wildcardtopic (which has a '*' and  a '?')
  std::cout << "Test 2 : Register a Listener to a Wildcardtopic(with a '*' in the middle)\n";
  try{
    sci::cca::WildcardTopic::pointer wildcardTopicPtr2(ptr->createWildcardTopic("t?st.*.?ne"));
    wildcardTopicPtr2->registerEventListener(std::string("WildcardTopicListener1"),evptr);
    
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a WildcardTopic : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }
  

  //  Register a Listener to a Wildcardtopic (with a '*' at the end)
  std::cout << "Test 3 : Register a Listener to a Wildcardtopic(with a '*' at the end)\n";
  try{
    sci::cca::WildcardTopic::pointer wildcardTopicPtr3(ptr->createWildcardTopic("test.*"));
    wildcardTopicPtr3->registerEventListener(std::string("WildcardTopicListener2"),evptr);
    
    std::cout << "Listener successfully registered\n";
  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "Exceptiion in registering a listener to a WildcardTopic : " << e->getNote() << std::endl; 
  }
  catch(const Exception &e){
    std::cout << "Unknown Exception\n";
  }


  //Send Events
  try{
     sci::cca::TypeMap::pointer eventHeader = com->getServices()->createTypeMap();
     
     sci::cca::TypeMap::pointer eventBody = com->getServices()->createTypeMap();
     eventBody->putString(std::string("Event1"),std::string("Event sent to randomTopic"));
     sci::cca::Event::pointer theEvent(new Event(eventHeader,eventBody));
     topicPtr->sendEvent(theEvent);

     sci::cca::TypeMap::pointer eventBody1 = com->getServices()->createTypeMap();
     eventBody1->putString(std::string("Event1"),std::string("Event sent to test.Echo.one"));
     sci::cca::Event::pointer theEvent1(new Event(eventHeader,eventBody1));
     topicPtr1->sendEvent(theEvent1);

     sci::cca::TypeMap::pointer eventBody2 = com->getServices()->createTypeMap();
     sci::cca::Event::pointer theEvent2(new Event(eventHeader,eventBody2));
     eventBody2->putString(std::string("Event2"),std::string("Event sent to test.Echo.two"));
     topicPtr2->sendEvent(theEvent2);

     sci::cca::TypeMap::pointer eventBody3 = com->getServices()->createTypeMap();
     eventBody3->putString(std::string("Event3"),std::string("Event sent to test.random.one"));
     sci::cca::Event::pointer theEvent3(new Event(eventHeader,eventBody3));
     topicPtr3->sendEvent(theEvent3);



  }
  catch(const sci::cca::EventServiceException::pointer &e){
    std::cout << "EventService Exception : " << e->getNote() << std::endl;
  }
  return 0;
}

