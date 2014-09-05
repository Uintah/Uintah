/**************************************
 * Test.h
 * Wayne Witzel
 * SCI Institute,
 * University of Utah
 *
 * Class Test:
 * Contains the name and results of a
 * regression test for reporting in a Suite
 * (see Suite.h).
 *
 ****************************************/

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
using namespace std;

class Test
{
public:
  Test(const string name);

  ~Test(); // delete space allocated for name

  const string getName()
  { return myName; }

  // set the results of the test:
  // hasBeenRun() will return true,
  // hasPassed() will return true only if setResults
  // has been given a true argument for the test each
  // time it as been called.
  void setResults(bool passed);

  // true iff this test has been run (setResults called).
  bool hasBeenRun()
  { return myHasBeenRun; }

  // true iff setResults has been called and it has been
  // given a true passed value each time it has been called.
  bool hasPassed()
  { return myHasPassed; }

private:
  string myName;

  // true iff this test has been run 
  // (that is, setResults has ever been called for this object)
  bool myHasBeenRun; 

  // true iff setResults has been called and it has been
  // given a true passed value each time it has been called.
  bool myHasPassed;  
};
