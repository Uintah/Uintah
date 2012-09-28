/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


/**************************************
 * Suite.h
 * Wayne Witzel
 * SCI Institute,
 * University of Utah
 *
 * Class Suite:
 * Manages and reports a suite of regression
 * tests.
 *
 ****************************************/

#include "Test.h"
//#include "StringCompare.h"
#include <map>
#include <list>


typedef std::map<std::string, Test*> testMap;
typedef std::map<std::string, Test*>::iterator testMapIterator;
typedef std::map<std::string, Test*>::value_type testMapPair;

class Suite
{
public:
  // construct a Suite with the given name
  Suite(const std::string& name);

  ~Suite(); // delete all created tests and myName

  const std::string getName() // get the Suite name
  { return myName; }

  // if the given testName has been used, return NULL;
  // otherwise, create a test with the given name,
  // add it to the suite, and return a pointer to it.
  Test* addTest(const std::string& testName);

  // Add the test and set its result.  If a test with
  // that name already exists, then simply return NULL.
  Test* addTest(const std::string& testName, bool result);

  // Same as addTest methods except that if the test name
  // already exists, that is what is used/returned instead of
  // NULL.
  Test* findOrAddTest(const std::string& testName);
  Test* findOrAddTest(const std::string& testName, bool results);
  
  // if there is a test in this suite with the given
  // name, return a pointer to it; otherwise return NULL.
  Test* findTest(const std::string& testName);

  bool hasAllPassed();

  bool hasAllBeenRun();
  
  // display a pass/fail report for all tests in this
  // suite along with a summary (to cout).
  void report();
private:
  std::string myName; // name of suite

  // map of tests (mapping names to tests)
  testMap myTests;
  std::list<Test*> myOrderedTests;
};

