#include "Test.h"

Test::Test(const string name)
: myName(name),
  myHasBeenRun(false),
  myHasPassed(false)
{
}

Test::~Test()
{
}

void Test::setResults(bool passed)
{
  if (myHasBeenRun == false) {
    myHasBeenRun = true;
    myHasPassed = passed;
  }
  else {
    // when you set the results many times, the actually
    // result is the logical and of all of these results.
    // (in other words, if it failed once, it failed in all)
    if (myHasPassed)
      myHasPassed = passed;
  }
}
