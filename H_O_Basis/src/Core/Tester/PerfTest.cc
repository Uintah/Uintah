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

#ifdef __digital__
#include <sys/types.h>
#define _BSD
#include <sys/timeb.h>
#endif

#include <Core/Tester/PerfTest.h>
#include <iostream>
#include <limits.h>
#include <string.h>
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

namespace SCIRun {

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
	    max=(int)(max*MINTIME/dt*1.2);
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

} // End namespace SCIRun

