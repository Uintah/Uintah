
/* REFERENCED */
static char *id="$Id$";

/*
 *  Time_unix.cc: Generic unix implementation of the Time class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/ThreadError.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syssgi.h>
#include <sys/time.h>

using SCICore::Thread::Time;

static int timer_32bit;
unsigned int iotimer_high;
volatile unsigned int* iotimer_addr32;
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
volatile unsigned int* iotimer_addr;
#define TIMERTYPE unsigned int
#else
volatile unsigned long *iotimer_addr;
#define TIMERTYPE unsigned long
#endif
static Time::SysClock orig_timer;
static double ticks_to_seconds;
static double seconds_to_ticks;
static int hittimer;
static bool initialized=false;
static SCICore::Thread::Mutex initlock("Time initialization lock");

#define TOPBIT ((unsigned int)0x80000000)

static
void
handle_alrm(int, int, sigcontext_t*)
{
    unsigned int t=*iotimer_addr32;
    unsigned int h=iotimer_high;
    if((t&TOPBIT) != (h&TOPBIT)){
	if((t&TOPBIT) == 0){
	    iotimer_high=(h&(~TOPBIT))+1;
	} else {
	    iotimer_high=h|TOPBIT;
	}
    }
    if(!hittimer)
	hittimer=1;
}

void
SCICore::Thread::Time::initialize()
{
    initlock.lock();
    if(initialized){
	initlock.unlock();
	return;
    }
    int poffmask = getpagesize() - 1;
    unsigned int cycleval;
    __psunsigned_t phys_addr = syssgi(SGI_QUERY_CYCLECNTR, &cycleval);
    __psunsigned_t raddr = phys_addr & ~poffmask;
    int fd = open("/dev/mmem", O_RDONLY);

    iotimer_addr = (volatile TIMERTYPE *)mmap(0, poffmask, PROT_READ,
					      MAP_PRIVATE, fd, (off_t)raddr);
    iotimer_addr = (volatile TIMERTYPE *)((__psunsigned_t)iotimer_addr +
					 (phys_addr & poffmask));
    iotimer_addr32 = (volatile unsigned int*)iotimer_addr;
    ticks_to_seconds=(double)cycleval*1.e-12;
    seconds_to_ticks=1./ticks_to_seconds;

    long ccsize=syssgi(SGI_CYCLECNTR_SIZE);
    if(ccsize == 32){
	timer_32bit=true;
    }

    double overflow=(65536.*65536.);
    if(!timer_32bit)
	overflow=overflow*overflow;
    overflow*=ticks_to_seconds;

    orig_timer=0;
    orig_timer=Time::currentTicks();

    iotimer_high=(*iotimer_addr32)&TOPBIT;

    if(timer_32bit){
	// Set up sigalrm handler...
	struct sigaction action;
	action.sa_flags=0;
	sigemptyset(&action.sa_mask);

	action.sa_handler=(SIG_PF)handle_alrm;
	if(sigaction(SIGALRM, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}

	int ticks=overflow/8;
	struct itimerval dt;
	dt.it_interval.tv_sec=ticks;
	dt.it_interval.tv_usec=0;
	dt.it_value.tv_sec=0;
	dt.it_value.tv_usec=1;
	struct itimerval old;
	if(setitimer(ITIMER_REAL, &dt, &old) != 0){
	    perror("setitimer");
	    exit(1);
	}
	while(!hittimer)
	    sigsuspend(0);
    }
    initialized=true;
    initlock.unlock();
}

Time::SysClock
SCICore::Thread::Time::currentTicks()
{
    if(!initialized)
	initialize();
    if(timer_32bit){
	for(;;){
	    unsigned high=iotimer_high;
	    unsigned ohigh=high;
	    unsigned low=*iotimer_addr32;
	    if((low&TOPBIT) != (high&TOPBIT)){
		// Possible rollover...
		if(!(low&TOPBIT))
		    high++;
	    }
	    if (ohigh == iotimer_high) {
		return ((long long)(high&(~TOPBIT))<<32|(long long)low)-orig_timer;
	    }
	    fprintf(stderr, "ROLLOVER loop around...\n");
	}
    } else {
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
	while (1) {
	    unsigned high = *iotimer_addr;
	    unsigned low = *(iotimer_addr + 1);
	    if (high == *iotimer_addr) {
		return ((long long)high<<32|(long long)low)-orig_timer;
	    }
	}
#else
	return *iotimer_addr-orig_timer;
#endif
    }
}

double
SCICore::Thread::Time::currentSeconds()
{
    return SCICore::Thread::Time::currentTicks()*ticks_to_seconds;
}

double
SCICore::Thread::Time::secondsPerTick()
{
    if(!initialized)
	initialize();
    return ticks_to_seconds;
}

double
SCICore::Thread::Time::ticksPerSecond()
{
    if(!initialized)
	initialize();
    return seconds_to_ticks;
}

void
SCICore::Thread::Time::waitUntil(double seconds)
{
    waitFor(seconds-currentSeconds());
}

void
SCICore::Thread::Time::waitFor(double seconds)
{
    if(!initialized)
	initialize();
    if(seconds<=0)
	return;
    static long tps=0;
    if(tps==0)
	tps=CLK_TCK;
    long ticks=(long)(seconds*(double)tps);
    while (ticks != 0){
	ticks=sginap(ticks);
    }
}

void
SCICore::Thread::Time::waitUntil(SysClock time)
{
    waitFor(time-currentTicks());
}

void
SCICore::Thread::Time::waitFor(SysClock time)
{
    if(!initialized)
	initialize();
    if(time<=0)
	return;
    static double tps=0;
    if(tps==0)
	tps=(double)CLK_TCK*ticks_to_seconds;
    int ticks=time*tps;
    while (ticks != 0){
	ticks=(int)sginap(ticks);
    }
}

//
// $Log$
// Revision 1.2  1999/08/25 22:53:42  sparker
// Fixed Time initialization race condition
//
// Revision 1.1  1999/08/25 22:36:55  sparker
// Broke out generic signal naming function into Thread_unix.{h,cc}
// Added irix hardware counter version of Time class in Time_irix.cc
//
// Revision 1.2  1999/08/25 19:00:53  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.1  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
