
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

#include <SCICore/share/share.h>

/*
 * Helper class for rigorous tests
 */

#define TEST(cond) __test->test(cond, #cond, __FILE__, __LINE__, __DATE__, __TIME__)

namespace SCICore {
namespace Tester {

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
// Revision 1.4  1999/07/06 20:53:24  moulding
// added share for win32 and modified timer stuff for portability
//
// Revision 1.3  1999/05/06 19:56:25  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:37  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
