
#include <Tester/TestTable.h>
#include <Classlib/String.h>
#include <Classlib/Queue.h>

TestTable test_table[] = {
    {"clString", &clString::test_rigorous, &clString::test_performance},
    {"Queue", Queue<int>::test_rigorous, 0},
    {0,0,0},
};

