/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "Suite.h"
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

// construct a Suite with the given name
Suite::Suite(const string& name)
: myName(name)
{ 
}

// delete all created tests and myName
Suite::~Suite()
{ 
  list<Test*>::iterator it = myOrderedTests.begin();
  for ( ; it != myOrderedTests.end(); it++) {
    delete *it;
  }
}

// if the given testName has been used, return NULL;
// otherwise, create a test with the given name,
// add it to the suite, and return a pointer to it.
Test* Suite::addTest(const string& testName)
{
  Test* newTest = new Test(testName);
  testMapPair name_test_pair(testName, newTest);
  bool insertResult = myTests.insert(name_test_pair).second;
  
  if (!insertResult) {
    // test name is already used, so cannot add
    cerr << "**ERROR**  Test name " << testName << " already used.";
    delete newTest;
    return NULL; 
  }

  myOrderedTests.push_back(newTest);
  return newTest; // new test added successfully
}

Test* Suite::addTest(const string& testName, bool result)
{
  Test* tst = addTest(testName);
  if (tst == NULL)
    return NULL;
  else {
    tst->setResults(result);
    return tst;
  }
}

Test* Suite::findOrAddTest(const string& testName)
{
  Test* foundTest = findTest(testName);
  if (foundTest != NULL)
    return foundTest;
  else {
    Test* newTest = new Test(testName);
    testMapPair name_test_pair(testName, newTest);
    assert(myTests.insert(name_test_pair).second);
    myOrderedTests.push_back(newTest);
    return newTest;
  }
}

Test* Suite::findOrAddTest(const string& testName, bool results)
{
  Test* tst = findOrAddTest(testName);
  tst->setResults(results);
  return tst;
}

// if there is a test in this suite with the given
// name, return a pointer to it; otherwise return NULL.
Test* Suite::findTest(const string& testName)
{
	// try to locate the test
  testMapIterator it = myTests.find(testName);

  if (it == myTests.end())
    return NULL; // test with given name doesn't exist
  else
    return (*it).second; // test found
}

// display a pass/fail report for all tests in this
// suite along with a summary (to cout).
void Suite::report()
{
  int num_passed = 0;
  int num_failed = 0;
  int num_not_run = 0;
  Test* test = NULL;

  cout << "=============================\n";
  cout << "Suite: " << myName << endl;
  cout << "-----------------------------\n\n";

  list<Test*>::iterator it = myOrderedTests.begin();
  for ( ; it != myOrderedTests.end(); it++) {
    test = (*it);
    if (test->hasBeenRun()) {
      if (test->hasPassed()) {
        cout << "Passed\t";
        num_passed++;
      }
      else {
        cout << "Failed\t";
        num_failed++;
      }
    }
    else {
      cout << "Not Run\t";
      num_not_run++;
    }
    cout << test->getName() << endl;
  }

  cout << "-----------------------------\n\n";

  cout << num_passed << " Passed\n";
  cout << num_failed << " Failed\n";
  cout << num_not_run << " Not Run\n";
}

bool Suite::hasAllPassed()
{
  list<Test*>::iterator it = myOrderedTests.begin();
  for ( ; it != myOrderedTests.end(); it++) {
    if (!(*it)->hasPassed())
      return false;
  }
  return true;
}

bool Suite::hasAllBeenRun()
{
  list<Test*>::iterator it = myOrderedTests.begin();
  for ( ; it != myOrderedTests.end(); it++) {
    if (!(*it)->hasBeenRun())
      return false;
  }
  return true;
}
