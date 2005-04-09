
#ifndef TESTER_TESTTABLE_H
#define TESTER_TESTTABLE_H 1

class RigorousTest;
class PerfTest;

struct TestTable {
    char* name;
    void (*rigorous_test)(RigorousTest*);
    void (*performance_test)(PerfTest*);
};

#endif
