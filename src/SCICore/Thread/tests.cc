
#include <Tester/TestTable.h>
#include <Thread/Thread.h>
#include <Thread/Time.h>

TestTable test_table[] = {
  {"Thread", &Thread::test_rigorous, 0},
  {"Time", &Time::test_rigorous, 0},
  {0,0,0}
};
