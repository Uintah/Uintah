
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

#include <SCICore/share/share.h>
#include <sys/timeb.h>

#define PERFTEST(testname) \
    __pt->start_test(testname); \
    while(__pt->do_test())

#define MINTIME 2.0
#define MULFACTOR 10

namespace SCICore {
namespace Tester {

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

} // End namespace Tester
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:20  mcq
// Initial commit
//
// Revision 1.4  1999/07/06 20:53:23  moulding
// added SHARE for win32 and modified timer stuff for portability
//
// Revision 1.3  1999/05/06 19:56:25  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:36  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
