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

#ifndef Framework_UnitTests_SmartPointerTest_h
#define Framework_UnitTests_SmartPointerTest_h

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

// The SmartPointer class is defined in Core/CCA/SmartPointer.h.
// There didn't seem to be much point in setting up a unit test directory
// in Core/CCA for a single class, so the SmartPointer tests reside here.

class SmartPointerTest : public CppUnit::TestFixture {
public:
  SmartPointerTest();
  virtual ~SmartPointerTest();

  // Set up context before running a test.
  virtual void setUp();

  // Clean up after the test run.
  virtual void tearDown();

protected:
  void testInstantiate();
  void testAddReference();

private:
  CPPUNIT_TEST_SUITE( SmartPointerTest );
  CPPUNIT_TEST(testInstantiate);
  CPPUNIT_TEST(testAddReference);
  CPPUNIT_TEST_SUITE_END();
};

#endif
