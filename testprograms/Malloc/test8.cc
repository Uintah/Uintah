
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

#include <iostream.h>

main()
{
    cerr << "This test not available\n";
    return 0;
}

#endif

