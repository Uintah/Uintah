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


#ifdef __sgi
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" int Allocator_try_lock(unsigned long*);

unsigned long lock;
int count;

void do_test(void*)
{
    for(int i=0;i<100000;i++){
//	fprintf(stderr, "trying lock...\n");
	while(!Allocator_try_lock(&lock)){
	    // spin...
//	    fprintf(stderr, "spinning (lock=%d)\n", lock);
	    sginap(0);
	}
	count++;
//	fprintf(stderr, "count=%d\n", count);
	lock=0;
    }
    fprintf(stderr, "count=%d\n", count);
}

main()
{
    sproc(do_test, PR_SADDR, 1);
    sproc(do_test, PR_SADDR, 2);
    sproc(do_test, PR_SADDR, 3);
    int s;
    wait(&s);
    wait(&s);
    wait(&s);
    return 0;
}

#else

#include <iostream>
using std::cerr;

int
main(char **, int )
{
    cerr << "This test not available\n";
    return 0;
}

#endif

