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

#include <Core/CCA/PIDL/PIDL.h>
#include <Framework/UnitTests/AbstractFrameworkTest.h>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( AbstractFrameworkTest );

AbstractFrameworkTest::AbstractFrameworkTest()
{
  //std::cout << "AbstractFrameworkTest::AbstractFrameworkTest()" << std::endl;
}

AbstractFrameworkTest::~AbstractFrameworkTest()
{
  //std::cout << "AbstractFrameworkTest::~AbstractFrameworkTest()" << std::endl;
}

void AbstractFrameworkTest::setUp()
{
  // tested in PIDLTest
  SCIRun::PIDL::initialize();
    //SCIRun::PIDL::isfrwk = true;
//   //all threads in the framework share the same
//   //invocation id
   //SCIRun::PRMI::setInvID(SCIRun::ProxyID(1,0));
//   fwk = new SCIRun::SCIRunFramework();
//   CPPUNIT_ASSERT(fwk != 0);
}

void AbstractFrameworkTest::tearDown()
{
  SCIRun::PIDL::finalize();
//   delete fwk;
}

void AbstractFrameworkTest::testInstantiate()
{
  CPPUNIT_FAIL( "not implemented" );
}

void AbstractFrameworkTest::testGetServices()
{
  CPPUNIT_FAIL( "not implemented" );
}
