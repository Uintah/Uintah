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
 *  tester.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Tester/PerfTest.h>
#include <Core/Tester/RigorousTest.h>
#include <Core/Tester/TestTable.h>
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Tester;

static bool passed=true;

using namespace SCIRun;

static void testrigorous(char* libname, char* classname)
{
    long start=PerfTest::time();

    void* handle=dlopen(libname, RTLD_LAZY);
    TestTable* table=(TestTable*)dlsym(handle, "test_table");
    if(!table){
	cerr << "Cannot find test in library: " << libname << '\n';
	exit(1);
    }
    int ntested=0;
    while(table->name){
	if(!classname || strcmp(table->name, classname)==0){
	    if(table->rigorous_test){
		cout << table->name << ": ";
		RigorousTest* pr=new RigorousTest(table->name);
		(*table->rigorous_test)(pr);
		if(pr->get_passed()){
		    cout << "passed (" << pr->get_ntests() << " tests)\n";
		} else {
		    cout << "FAILED " << pr->get_nfailed() << " of " << pr->get_ntests() << " tests\n";
		    passed=false;
		}
	    }
	    ntested++;
	}
	table++;
    }
    if(ntested==0){
	cerr << "No tests found!\n";
	exit(1);
    }
    if(!passed){
	cerr << "Rigorous tests FAILED!\n";
	cout.flush();
	exit(1);
    }
    long stop=PerfTest::time();
    cout << ntested << " rigorous tests passed in " << PerfTest::deltat(stop-start) << " seconds\n";
}

static void usage(char * progname)
{
    cerr << "Usage: " << progname << " libs ....\n";
    cerr << "Please specify at least one of:\n";
    cerr << "\t-rigorous\tperform rigorous tests\n";
    cerr << "\t-perf\tdo performance measurements\n";
    exit(1);
}

/*
 * Run performance tests on <tt>libname</tt>.
 */
static void testperf(char* libname, char* classname)
{
    void* handle=dlopen(libname, RTLD_LAZY);
    TestTable* table=(TestTable*)dlsym(handle, "test_table");
    if(!table){
	cerr << "Cannot find test in library: " << libname << '\n';
    }
    int ntested=0;
    while(table->name){
	if(!classname || strcmp(table->name, classname)==0){
	    if(table->performance_test){
		int l=strlen(table->name);
		int n1=(72-2-l)/2;
		int n2=72-2-l-n1;
		while(n1--)
		    cout << '-';
		cout << ' ' << table->name << ' ';
		while(n2--)
		    cout << '-';
		cout << '\n';
		PerfTest* pr=new PerfTest(table->name);
		(*table->performance_test)(pr);
		pr->finish();
		ntested++;
	    }
	}
	table++;
    }
}

int main(int argc, char* argv[])
{
    char** libs=new char*[argc];
    int nlibs=0;
    bool do_testperf=false;
    bool do_testrig=false;
    char* classname=0;
    for(int i=1;i<argc;i++){
	if(argv[i][0]=='-'){
	    if(strcmp(argv[i], "-perf") == 0){
		do_testperf=true;
	    } else if(strcmp(argv[i], "-rigorous") == 0){
		do_testrig=true;
	    } else if(strcmp(argv[i], "-class") == 0){
		i++;
		if(i>=argc){
		    usage(argv[0]);
		}
		classname=argv[i];
	    } else {
		cerr << "Unknown option: " << argv[i] << endl;
		exit(1);
	    }
	} else {
	    libs[nlibs++]=argv[i];
	}
    }
    if(!do_testrig && !do_testperf){
	usage(argv[0]);
    }
    if(nlibs==0){
	usage(argv[0]);
    }
    
    passed=true;
    if(do_testrig){
	for(int i=0;i<nlibs;i++)
	    testrigorous(libs[i], classname);
    }
    cout.flush();
    if(!passed){
	cerr << "Failed rigorous tests, quitting now\n";
	exit(1);
    }
    if(do_testperf){
	for(int i=0;i<nlibs;i++)
	    testperf(libs[i], classname);
    }
    cout.flush();
    if(!passed){
	cerr << "Failed performance tests?\n";
	exit(1);
    }
    return 0;
}

