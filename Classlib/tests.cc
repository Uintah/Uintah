
#include <Tester/TestTable.h>
#include <Classlib/String.h>
#include <Classlib/Queue.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/Array3.h>
#include <Classlib/HashTable.h>

TestTable test_table[] = {
    {"clString", &clString::test_rigorous, &clString::test_performance},
    {"Queue", Queue<int>::test_rigorous, 0},
    {"Array1", Array1<float>::test_rigorous, 0},
    {"Array2", Array2<int>::test_rigorous, 0},
    {"Array3", Array3<int>::test_rigorous, 0},
    {"HashTable", HashTable<char*, int>::test_rigorous, 0},
    {0,0,0}
};

