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


#include<CCA/Components/TestCreateTopic/TestCreateTopic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestCreateTopic()
{
  return sci::cca::Component::pointer(new TestCreateTopic());
}


TestCreateTopic::TestCreateTopic()
{
}

TestCreateTopic::~TestCreateTopic()
{
  services->removeProvidesPort("go");
}

void TestCreateTopic::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestExecute *providesgp = new TestExecute();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestExecute::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);
}

int TestExecute::go()
{
  std::cout << "Running EventService::createTopic tests:" << std::endl;

  sci::cca::ports::EventService::pointer eventServicePort;
  try {
    sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
    eventServicePort = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
    if (eventServicePort.isNull()) {
      std::cout << "Pointer returned from getPort is Null!!!\n";
      return 1;
    }
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cout << e->getNote() << std::endl;
    return 1;
  }

  topicEmptyName(eventServicePort);
  topicReturnsNullPointer(eventServicePort);
  topicReturnsExistingPointer(eventServicePort);

  com->getServices()->releasePort("cca.EventService");
  return 0;
}

void TestExecute::topicEmptyName(const sci::cca::ports::EventService::pointer& eventServicePort)
{
  //Creating a topic with an empty topic name
  std::cout << "Create Topic with an empty name\n";
  try {
    sci::cca::Topic::pointer topicPtr = eventServicePort->createTopic(testEmptyTopicName);
  }
  catch(const sci::cca::EventServiceException::pointer &e) {
    std::cout << "Caught exception: " << e->getNote() << std::endl;
  }
}

void TestExecute::topicReturnsNullPointer(const sci::cca::ports::EventService::pointer& eventServicePort)
{
   //Check if createTopic return a null pointer
   std::cout << "Test 2: Check if createTopic return a null pointer\n";

   std::string testTopicName("TestExecute::topicReturnsNullPointer");

   sci::cca::Topic::pointer samplePtr = eventServicePort->createTopic(testTopicName);
   if (samplePtr.isNull())
     std::cout << "Pointer returned from Create Topic is Null\n";
   else
     std::cout << "Pointer returned from Create Topic is not Null\n";
}

void TestExecute::topicReturnsExistingPointer(const sci::cca::ports::EventService::pointer& eventServicePort)
{
   //Check if createTopic with an already existing topicname returns a pointer to that existing topic
   std::cout << "Test 3: Check if createTopic with an already existing topicname returns a pointer to that existing topic\n";

   std::string testTopicName("TestExecute::topicReturnsExistingPointer");

   sci::cca::Topic::pointer samplePtr = eventServicePort->createTopic(testTopicName);
   sci::cca::Topic::pointer duplicateTopicPtr = eventServicePort->createTopic(testTopicName);
   if (samplePtr == duplicateTopicPtr)
     std::cout << "Same pointer returned for duplicate topic\n";
   else
     std::cout << "Different  pointer returned for duplicate topic\n";
}
