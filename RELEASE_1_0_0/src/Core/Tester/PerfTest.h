/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
