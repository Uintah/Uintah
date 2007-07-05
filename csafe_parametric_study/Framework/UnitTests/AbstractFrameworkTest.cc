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

#include <Framework/UnitTests/AbstractFrameworkTest.h>
#include <Framework/SCIRunFramework.h>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( AbstractFrameworkTest );

AbstractFrameworkTest::AbstractFrameworkTest()
{
}

AbstractFrameworkTest::~AbstractFrameworkTest()
{
}

void AbstractFrameworkTest::setUp()
{
  SCIRun::SCIRunFramework *fwk = new SCIRun::SCIRunFramework();
  CPPUNIT_ASSERT_MESSAGE("framework pointer is not null", fwk != 0);
  abstractFramework = sci::cca::AbstractFramework::pointer(fwk);
  CPPUNIT_ASSERT_MESSAGE("verify smart pointer", abstractFramework.getPointer() == fwk);
}

void AbstractFrameworkTest::tearDown()
{
  // Let smart pointer do the cleanup.
}

void AbstractFrameworkTest::testInstantiate()
{
  // testing abstract framework instantiated in setUp
  CPPUNIT_ASSERT_MESSAGE("AbstractFramework smart pointer is not null", ! abstractFramework.isNull());
  CPPUNIT_ASSERT(abstractFramework.isUnique());
}

void AbstractFrameworkTest::testCreateTypeMap()
{
  sci::cca::TypeMap::pointer typeMap = abstractFramework->createTypeMap();
  CPPUNIT_ASSERT_MESSAGE("TypeMap smart pointer is not null", ! typeMap.isNull());
  CPPUNIT_ASSERT(typeMap.isUnique());
  //sci::cca::TypeMap::pointer typeMap2 = typeMap;
  //CPPUNIT_ASSERT(typeMap.isUnique());
  //CPPUNIT_ASSERT(typeMap2.isUnique());
}

void AbstractFrameworkTest::testGetServices()
{
  sci::cca::TypeMap::pointer typeMap = abstractFramework->createTypeMap();
  sci::cca::Services::pointer services = abstractFramework->getServices("testAbstractFramework", "AbstractFramework", typeMap);
  CPPUNIT_ASSERT_MESSAGE("Services smart pointer is not null", ! services.isNull());

  // should Services object be unique??? - there's an addReference call in CCAComponentModel::createServices...
  //CPPUNIT_ASSERT(services.isUnique());
}

void AbstractFrameworkTest::testReleaseServices()
{
  sci::cca::TypeMap::pointer typeMap = abstractFramework->createTypeMap();
  sci::cca::Services::pointer services = abstractFramework->getServices("testAbstractFramework", "AbstractFramework", typeMap);
  CPPUNIT_ASSERT_MESSAGE("Services smart pointer is not null", ! services.isNull());
  abstractFramework->releaseServices(services);
}

// Really simple test, since few threads are running (DataTransmitter threads and main)
// and no components have been created...
// AbstractFramework doesn't have any kind of state monitoring...
void AbstractFrameworkTest::testShutdownFramework()
{
  std::cout << "\nAbstractFrameworkTest::testShutdownFramework()" << std::endl;
  abstractFramework->shutdownFramework();
  std::cout << "\nAbstractFrameworkTest::testShutdownFramework() done" << std::endl;
}
