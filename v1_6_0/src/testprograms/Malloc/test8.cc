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

void do_test(void*)
{
    for(int i=0;i<200000;i++){
	void* p=malloc(508);
	free(p);
    }
    fprintf(stderr, "done...\n");
}

main()
{
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    int s;
    wait(&s);
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

