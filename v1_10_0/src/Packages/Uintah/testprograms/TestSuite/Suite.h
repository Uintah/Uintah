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

typedef map<string, Test*> testMap;
typedef map<string, Test*>::iterator testMapIterator;
typedef map<string, Test*>::value_type testMapPair;

class Suite
{
public:
  // construct a Suite with the given name
  Suite(const string& name);

  ~Suite(); // delete all created tests and myName

  const string getName() // get the Suite name
  { return myName; }

  // if the given testName has been used, return NULL;
  // otherwise, create a test with the given name,
  // add it to the suite, and return a pointer to it.
  Test* addTest(const string& testName);

  // Add the test and set its result.  If a test with
  // that name already exists, then simply return NULL.
  Test* addTest(const string& testName, bool result);

  // Same as addTest methods except that if the test name
  // already exists, that is what is used/returned instead of
  // NULL.
  Test* findOrAddTest(const string& testName);
  Test* findOrAddTest(const string& testName, bool results);
  
  // if there is a test in this suite with the given
  // name, return a pointer to it; otherwise return NULL.
  Test* findTest(const string& testName);

  bool hasAllPassed();

  bool hasAllBeenRun();
  
  // display a pass/fail report for all tests in this
  // suite along with a summary (to cout).
  void report();
private:
  string myName; // name of suite

  // map of tests (mapping names to tests)
  testMap myTests;
  list<Test*> myOrderedTests;
};

