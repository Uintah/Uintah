
#include <Tester/TestTable.h>
#include <Classlib/String.h>

TestTable test_table[] = {
    {"clString", &clString::test_rigorous, &clString::test_performance},
    {0,0,0},
};

