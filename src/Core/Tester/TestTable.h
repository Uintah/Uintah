
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

namespace SCICore {
namespace Tester {

class RigorousTest;
class PerfTest;

struct TestTable {
    char* name;
    void (*rigorous_test)(RigorousTest*);
    void (*performance_test)(PerfTest*);
};

} // End namespace Tester
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:20  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:26  dav
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
