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
 *  RigorousTest.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef TESTER_RIGOROUSTEST_H
#define TESTER_RIGOROUSTEST_H 1

#include <Core/share/share.h>

/*
 * Helper class for rigorous tests
 */

#define TEST(cond) __test->test(cond, #cond, __FILE__, __LINE__, __DATE__, __TIME__)

namespace SCIRun {

class SCICORESHARE RigorousTest {
public:
    RigorousTest(char* symname);
    ~RigorousTest();
    void test(bool condition, char* descr, char* file, int line, char* date, char* time);

    bool get_passed();
    int get_ntests();
    int get_nfailed();

private:
    char* symname;

    bool passed;
    int ntests;
    int nfailed;
    int nprint;
    char* old_descr;
    char* old_file;
    int old_line;
    char* old_date;
    char* old_time;
};

} // End namespace SCIRun


#endif
