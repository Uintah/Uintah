//static char *id="@(#) $Id$";

/*
 *  RigorousTest.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Tester/RigorousTest.h>
#include <iostream.h>

namespace SCICore {
namespace Tester {

RigorousTest::RigorousTest(char* symname)
    : symname(symname)
{
    passed=true;
    ntests=nfailed=0;
    nprint=0;
    old_descr=old_file=old_date=old_time=0;
    old_line=-1;
}
RigorousTest::~RigorousTest()
{
}

void RigorousTest::test(bool condition, char* descr, char* file,
			int line, char* date, char* time)
{
    ntests++;
    if(!condition) {
	nfailed++;
	passed=false;
	if(nprint < 25){
	    if(descr != old_descr || file != old_file || line != old_line || date != old_date || time != old_time) {
		if(nprint==0)
		    cout << "\n";
		cout << '"' << file << "\", line " << line << ":\n";
		cout << "Test FAILED: \"" << descr << "\"\n";
		cout << "Compiled: " << date << ' ' << time << "\n\n";
		nprint++;
		old_descr=descr;
		old_file=file;
		old_line=line;
		old_date=date;
		old_time=time;
		if(nprint == 25)
		    cout << "More than 25 failures - no more will be reported for this class\n";
	    }
	}
    }
}

int RigorousTest::get_nfailed()
{
    return nfailed;
}

bool RigorousTest::get_passed()
{
    return passed;
}

int RigorousTest::get_ntests()
{
    return ntests;
}

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
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
