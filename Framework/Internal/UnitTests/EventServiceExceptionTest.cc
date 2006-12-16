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

#include <Framework/Internal/UnitTests/EventServiceExceptionTest.h>
#include <Framework/Internal/EventServiceException.h>
#include <Framework/Internal/Topic.h>
#include <Framework/SCIRunFramework.h>

#include <iostream>
#include <sstream>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( EventServiceExceptionTest );

EventServiceExceptionTest::EventServiceExceptionTest()
{
}

EventServiceExceptionTest::~EventServiceExceptionTest()
{
}

void EventServiceExceptionTest::setUp()
{
}

void EventServiceExceptionTest::tearDown()
{
}

void EventServiceExceptionTest::testInstantiate()
{
  SCIRun::EventServiceExceptionPtr exEmptyMessage(new SCIRun::EventServiceException(""));
  CPPUNIT_ASSERT_MESSAGE("EventServiceException with empty message smart pointer is not null", ! exEmptyMessage.isNull());

  SCIRun::EventServiceExceptionPtr exWithMessage(new SCIRun::EventServiceException("EventServiceException test message"));
  CPPUNIT_ASSERT_MESSAGE("EventServiceException with message smart pointer is not null", ! exWithMessage.isNull());

  SCIRun::EventServiceExceptionPtr exWithMessageAndType(new SCIRun::EventServiceException("EventServiceException with type test message", sci::cca::NetworkError));
  CPPUNIT_ASSERT_MESSAGE("EventServiceException with message and type smart pointer is not null", ! exWithMessageAndType.isNull());
}

void EventServiceExceptionTest::testSetNote()
{
  CPPUNIT_FAIL("Not implemented yet");
}

void EventServiceExceptionTest::testAdd()
{
  CPPUNIT_FAIL("Not implemented yet");
}

void EventServiceExceptionTest::testThrowAndCatch()
{
  struct ExceptionThrower {
    void exceptionThrow() {
      throw SCIRun::EventServiceExceptionPtr(new SCIRun::EventServiceException("EventServiceException throw and catch test message"));
    }
  } exceptionThrower;

  try {
    exceptionThrower.exceptionThrow();
  }
  catch (SCIRun::EventServiceExceptionPtr ex) {
    std::cout << "\nEventServiceExceptionTest::testThrowAndCatch() caught exception: " << ex->getNote() << std::endl;
    std::cout << "\nEventServiceExceptionTest::testThrowAndCatch() get exception trace: " << ex->getTrace() << std::endl;
    CPPUNIT_ASSERT(ex->getCCAExceptionType() == sci::cca::Nonstandard);
  }
}

void EventServiceExceptionTest::testThrowAndCatchWithType()
{
  CPPUNIT_FAIL("Not implemented yet");
}
