
#include <Tester/TestTable.h>
#include <Classlib/String.h>
#include <Classlib/Queue.h>
#include <Classlib/Array1.h>

TestTable test_table[] = {
    {"clString", &clString::test_rigorous, &clString::test_performance},
    {"Queue", Queue<int>::test_rigorous, 0},
    {"Array1", Array1<int>::test_rigorous, 0},
    {0,0,0}
};

