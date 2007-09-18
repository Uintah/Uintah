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

#include <Framework/UnitTests/BuilderServiceTest.h>
#include "sidl.hxx"

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( BuilderServiceTest );

BuilderServiceTest::BuilderServiceTest()
{
}

BuilderServiceTest::~BuilderServiceTest()
{
}

void BuilderServiceTest::setUp()
{
  CPPUNIT_ASSERT(sjf._is_nil());
  CPPUNIT_ASSERT(framework._is_nil());
  CPPUNIT_ASSERT(ccaFramework._is_nil());

  sjf = scijump::SCIJumpFramework::_create();
  //CPPUNIT_ASSERT_MESSAGE("Created scijump::SCIJumpFramework is not null", sjf._not_nil());
  //framework = ::sidl::babel_cast<sci::cca::BuilderService>(sjf);
  //ccaFramework = ::sidl::babel_cast<gov::cca::AbstractFramework>(sjf);
}

void BuilderServiceTest::tearDown()
{
  // Let Babel do the cleanup.
}
