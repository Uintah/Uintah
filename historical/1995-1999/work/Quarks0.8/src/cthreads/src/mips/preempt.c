/* 
 * File:	preempt.c
 * Description: 
 * Author:	Leigh Stoller
 * 		Computer Science Dept.
 * 		University of Utah
 * Date:	17-Nov-90
 *
 * (c) Copyright 1990, University of Utah, all rights reserved.
 * 
 */


#include <sys/signal.h> /* Berkeley */
#include <sys/time.h> 

#include "cthreads.h"

/* Prototypes for forward declarations */
void sigrelease(int sig);
void settimer(int usecs);


static int clock_int = 10000;		/* In usecs */

static int block_clock = 0;

void splhigh(void)
{
	block_clock = 1;
}

void spllow(void)
{
	block_clock = 0;
}

int hardclock(int dummy)
{
	if (!block_clock) {
	    sigrelease(SIGVTALRM);
	    cthread_yield();
	}
        return 0;

}

void stopclock(void)
{
	settimer(0);
}

void startclock(void)
{
	struct sigvec vec, ovec;

	vec.sv_handler = (int (*)(int)) hardclock;
	vec.sv_mask    = 0;
	vec.sv_flags   = 0;
  
	if (sigvec(SIGVTALRM, &vec, &ovec) < 0) {
		perror("Could not set signal handler for hardclock.");
		exit(1);
	}
	settimer(clock_int);
}

int
setclockrate(int rate)
{
	int oldrate = clock_int;
	
	clock_int = rate;
	startclock();
	return(oldrate);
}

void
settimer(int usecs)
{
	struct itimerval timer;
	
	timer.it_value.tv_sec  = timer.it_interval.tv_sec  = 0;
	timer.it_value.tv_usec = timer.it_interval.tv_usec = usecs;
	setitimer(ITIMER_VIRTUAL, &timer, NULL);
}

/*
 * Unblock the received signal.
 */
void 
sigrelease(int sig)
{
	sigsetmask(sigblock(0) & ~sigmask(sig));
}




