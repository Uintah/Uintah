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

#include <Core/Tester/RigorousTest.h>
#include <iostream>
using std::cout;

namespace SCIRun {

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

} // End namespace SCIRun

