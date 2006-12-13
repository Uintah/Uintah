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

#include <Core/CCA/PIDL/UnitTests/PIDLTest.h>

#include <cppunit/TestAssert.h>
#include <cppunit/portability/Stream.h>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/Warehouse.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/CCA/PIDL/CommError.h>

#include <iostream>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( PIDLTest );

PIDLTest::~PIDLTest()
{
}

void PIDLTest::setUp()
{
}

void PIDLTest::tearDown()
{
}

void PIDLTest::testInitialize()
{
  try {
    SCIRun::PIDL::initialize();
  }
  catch (const SCIRun::CommError& e1) {
    CPPUNIT_FAIL(e1.message());
  }
  catch (const SCIRun::Exception& e2) {
    CPPUNIT_FAIL(e2.message());
  }
  // check initial PIDL settings
  CPPUNIT_ASSERT_EQUAL(SCIRun::PIDL::rank, 0);
  CPPUNIT_ASSERT_EQUAL(SCIRun::PIDL::size, 1);
  CPPUNIT_ASSERT(SCIRun::PIDL::sampleProxy == false);
  CPPUNIT_ASSERT(SCIRun::PIDL::isfrwk == false);
  CPPUNIT_ASSERT(SCIRun::PIDL::isFramework() == false);
}

void PIDLTest::testWarehouse()
{
  SCIRun::Warehouse *warehouse;
  try {
    warehouse = SCIRun::PIDL::getWarehouse();
  }
  catch (const SCIRun::InternalError& e) {
    CPPUNIT_FAIL(e.message());
  }
  CPPUNIT_ASSERT(warehouse != 0);
}

void PIDLTest::testFinalize()
{
  SCIRun::PIDL::finalize();
}
