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

#include <Core/CCA/PIDL/UnitTests/SmartPointerTest.h>
#include <Core/CCA/SmartPointer.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>

#include <iostream>
#include <sstream>

// In the SCIJump framework, SmartPointer objects are used with Objects
// (see Core/CCA/PIDL/Object.h).
// To simplify testing SmartPointers, a simplified version of Object
// is provided here:
class TestObject {
public:
  TestObject() : refCount(0)
  {
    om = new SCIRun::Mutex("TestObject mutex");
  }
  ~TestObject()
  {
    delete om;
  }
  const int getReferenceCount() const { return refCount; }
  void _addReference()
  {
    SCIRun::Guard g(om);
    refCount++;
  }
  void _deleteReference()
  {
    {
      SCIRun::Guard g(om);
      refCount--;
    }

    bool del = false;
    if (refCount == 0) {
      del = true;
    }
    if (del) {
      delete this;
    }
  }

private:
  int refCount;
  SCIRun::Mutex* om;
};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( SmartPointerTest );

SmartPointerTest::SmartPointerTest()
{
}

SmartPointerTest::~SmartPointerTest()
{
}

void SmartPointerTest::setUp()
{
}

void SmartPointerTest::tearDown()
{
}

void SmartPointerTest::testInstantiate()
{
  CCALib::SmartPointer<TestObject> testObjectPtr;
  CPPUNIT_ASSERT(testObjectPtr.useCount() == 0);

  testObjectPtr = CCALib::SmartPointer<TestObject>(new TestObject);
  CPPUNIT_ASSERT(testObjectPtr.useCount() == 1);
}

void SmartPointerTest::testAddReference()
{
  CCALib::SmartPointer<TestObject> testObjectPtr(new TestObject);
  CPPUNIT_ASSERT(testObjectPtr.useCount() == 1);

  testObjectPtr.reference();
  CPPUNIT_ASSERT(testObjectPtr.useCount() == 2);
}
