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


#include<CCA/Components/TestGetTopic/TestGetTopic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestGetTopic()
{
  return sci::cca::Component::pointer(new TestGetTopic());
}


TestGetTopic::TestGetTopic()
{
}

TestGetTopic::~TestGetTopic()
{
  services->removeProvidesPort("go");
}

void TestGetTopic::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestGetTopicgo1 *providesgp = new TestGetTopicgo1();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestGetTopicgo1::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestGetTopicgo1::go()
{
  std::cout<<"Inside Go Function of TestGetTopic\n";
   sci::cca::ports::EventService::pointer ptr;
   try {
      sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
      ptr = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
      if(ptr.isNull())
	std::cout << "Pointer returned from getPort is Null!!!\n";
   }
    catch (const sci::cca::CCAException::pointer &e) {
      std::cout << e->getNote() << std::endl;
    return 1;
  }
   //calling getTopic for a topic that doesn't exist
   std::cout << "Test 1: Calling getTopic for a topic that doesn't exist:\n";
   try
     {
       sci::cca::Topic::pointer randomPtr = ptr->getTopic(std::string("randomString"));
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in calling getTopic for a topic that doesn't exist: " << e->getNote() << std::endl;
     }
   std::string testString("Sample Topic");
   sci::cca::Topic::pointer samplePtr = ptr->createTopic(testString);
   sci::cca::Topic::pointer topicPtr = ptr->getTopic(testString);

   //Check if pointer returned from getTopic is null
   std::cout << "Test 2: Check if pointer returned from getTopic is null:\n";
   if(topicPtr.isNull())
     std::cout << "Pointer returned from getTopic is Null\n";
   else
     std::cout << "Pointer returned from getTopic is not Null\n";
   
   //check if getTopic returns the same pointer as createTopic
   std::cout << "Test 3 : Check if getTopic returns the same pointer as createTopic:\n";
   if(samplePtr == topicPtr)
     std::cout << "Same pointer returned for getTopic\n";
   else
     std::cout << "Different  pointer returned for getTopic\n";
   return 0;
}

