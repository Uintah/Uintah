
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

main()
{
    cerr << "This test not available\n";
    return 0;
}

#endif

