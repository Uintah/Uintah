
/*
 *  TestTable.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef TESTER_TESTTABLE_H
#define TESTER_TESTTABLE_H 1

namespace SCIRun {

class RigorousTest;
class PerfTest;

struct TestTable {
    char* name;
    void (*rigorous_test)(RigorousTest*);
    void (*performance_test)(PerfTest*);
};

} // End namespace SCIRun


#endif
