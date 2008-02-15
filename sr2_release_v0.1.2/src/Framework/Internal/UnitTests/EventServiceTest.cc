/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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

#include <Framework/Internal/UnitTests/EventServiceTest.h>
#include <Framework/Internal/EventService.h>
#include <Framework/Internal/EventServiceException.h>
#include <Framework/Internal/Topic.h>
#include <Framework/SCIRunFramework.h>

#include <iostream>
#include <sstream>
#include <stack>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( EventServiceTest );

EventServiceTest::EventServiceTest()
{
  for (int i = 0; i < MAX_TEST_TOPICS; i++) {
    std::ostringstream o;
    o << "topic.test.name" << i+1 << std::endl;
    testTopicNames.push_back(o.str());
  }
}

EventServiceTest::~EventServiceTest()
{
  testTopicNames.clear();
}

void EventServiceTest::setUp()
{
  // The following code is tested in Framework/UnitTests/...
  SCIRun::SCIRunFramework *fwk = new SCIRun::SCIRunFramework();
  CPPUNIT_ASSERT_MESSAGE("framework pointer is not null", fwk != 0);

  abstractFramework = sci::cca::AbstractFramework::pointer(fwk);
  CPPUNIT_ASSERT_MESSAGE("verify smart pointer", abstractFramework.getPointer() == fwk);

  services = abstractFramework->getServices("testAbstractFramework", "AbstractFramework",
    abstractFramework->createTypeMap());
  CPPUNIT_ASSERT_MESSAGE("Services smart pointer is not null", ! services.isNull());

  eventServicePort =
    pidl_cast<sci::cca::ports::EventService::pointer>(services->getPort("cca.EventService"));
}

void EventServiceTest::tearDown()
{
  services->releasePort("cca.EventService");
  abstractFramework->releaseServices(services);
}

void EventServiceTest::testInstantiate()
{
  // testing abstract framework instantiated in setUp
  CPPUNIT_ASSERT_MESSAGE("EventService Port smart pointer is not null", ! eventServicePort.isNull());
  std::cout << "\nEventService Port use count=" << eventServicePort.useCount() << std::endl;
  //CPPUNIT_ASSERT(eventServicePort.isUnique());
}

void EventServiceTest::testCreateTopicWithEmptyString()
{
  sci::cca::Topic::pointer topic;
  try {
    topic = eventServicePort->createTopic("");
  }
  catch (SCIRun::EventServiceExceptionPtr ex) {
    std::cout << "\nEventServiceExceptionTest::testCreateTopicWithEmptyString() caught exception: " << ex->getNote() << std::endl;
  }
    CPPUNIT_ASSERT_MESSAGE("Topic smart pointer is null", topic.isNull());
}

void EventServiceTest::testCreateTopic()
{
  std::stack<sci::cca::Topic::pointer> topics;
  for (int i = 0; i < MAX_TEST_TOPICS; i++) {
    topics.push(eventServicePort->createTopic(testTopicNames[i]));
  }

  while (! topics.empty()) {
    sci::cca::Topic::pointer topic = topics.top();
    CPPUNIT_ASSERT_MESSAGE("Topic smart pointer is not null", ! topic.isNull());
    topics.pop();
  }
}
void EventServiceTest::testCreateTopicBadNameFormat()
{
  CPPUNIT_FAIL("Not implemented yet");
}

void EventServiceTest::testGetTopic()
{
  for (int i = 0; i < MAX_TEST_TOPICS; i++) {
    eventServicePort->createTopic(testTopicNames[i]);
  }

  for (int i = 0; i < MAX_TEST_TOPICS; i++) {
    sci::cca::Topic::pointer topic = eventServicePort->getTopic(testTopicNames[i]);
    CPPUNIT_ASSERT_MESSAGE("Topic smart pointer is not null", ! topic.isNull());
    CPPUNIT_ASSERT_MESSAGE("Topic name set", topic->getTopicName() == testTopicNames[i]);
  }
}

