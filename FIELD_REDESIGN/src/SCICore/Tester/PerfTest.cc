//static char *id="@(#) $Id$";

/*
 *  PerfTest.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Tester/PerfTest.h>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <iostream>
using std::cout;
using std::endl;

#ifndef _WIN32
  #include <unistd.h>
#endif

#ifdef LINUX
  #include <asm/param.h>
#elif !defined(_WIN32)
  #include <sys/param.h>
#endif

namespace SCICore {
namespace Tester {

PerfTest::PerfTest(char* symname)
    : symname(symname)
{
    baseline=0;
}

PerfTest::~PerfTest()
{
}
void PerfTest::print_msg()
{
    cout << max;
    cout.flush();
}
void PerfTest::start_test(char* name)
{
    cout << "\nStarting test: " << name << endl;
    count=0;
    max=1;
    print_msg();
    ftime(&start_time);
    if(strcmp(name, "baseline") == 0)
	is_baseline=true;
    else
	is_baseline=false;
    if(strcmp(name, "sanity") == 0)
	is_sanity=true;
    else
	is_sanity=false;
}
bool PerfTest::do_test()
{
    if(count++<max)
	return true;

	struct timeb stop_time;
	ftime(&stop_time);
	double dt = (stop_time.time-start_time.time)*1000 + (stop_time.millitm-start_time.millitm);
    cout << '(' << dt/**1000*/ << " ms)";
    if(dt < MINTIME) {
	// Continue the tests...
	if(dt == 0)
	    max*=MULFACTOR;
	else if(MINTIME/dt > MULFACTOR)
	    max*=MULFACTOR;
	else
	    max*=MINTIME/dt*1.2;
	count=0;
	if(dt==0)
	    cout << '\r';
	else
	    cout << ' ';
	print_msg();
	ftime(&start_time);
	return true;
    }
    dt/=max;
    if(is_baseline)
	baseline=dt;
    else
	dt-=baseline;
    cout << "\n ->" << dt*1000 << " ms/operation";
    if(!is_baseline)
	cout << " above baseline\n";
    else
	cout << " baseline\n";
    return false;
}

void PerfTest::finish()
{
}

void PerfTest::time(struct timeb* t)
{
	ftime(t);
}

} // End namespace Tester
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/10/07 02:08:05  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:39:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:19  mcq
// Initial commit
//
// Revision 1.3  1999/07/06 20:53:24  moulding
// added SHARE for win32 and modified timer stuff for portability
//
// Revision 1.2  1999/06/30 21:43:37  dav
// updated for linux
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
