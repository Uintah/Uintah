
/*
 *  PerfTest.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef TESTER_PERFTEST_H
#define TESTER_PERFTEST_H 1

#include <Core/share/share.h>
#include <sys/timeb.h>

#define PERFTEST(testname) \
    __pt->start_test(testname); \
    while(__pt->do_test())

#define MINTIME 2.0
#define MULFACTOR 10

namespace SCIRun {

class SCICORESHARE PerfTest {
public:
    PerfTest(char* symname);
    ~PerfTest();
    void start_test(char* name);
    bool do_test();
    void finish();

    static void time(struct timeb*);
    static double deltat(long ticks);
private:
    void print_msg();
    char* symname;
    int count;
    int max;
    struct timeb start_time;
    double baseline;
    double baseline_time;
    bool is_baseline;
    bool is_sanity;
};

} // End namespace SCIRun


#endif
