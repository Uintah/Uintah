/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Time_unix.cc: Generic unix implementation of the Time class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 */


#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadError.h>
#include <fcntl.h>
#include <signal.h>
#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syssgi.h>
#include <sys/time.h>
#include <cerrno>

using namespace SCIRun;

static int timer_32bit;
unsigned int iotimer_high;
volatile unsigned int* iotimer_addr32;
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
volatile unsigned int* iotimer_addr;
#define TIMERTYPE unsigned int
#else
volatile unsigned long long *iotimer_addr;
#define TIMERTYPE unsigned long long
#endif
static Time::SysClock orig_timer;
static double ticks_to_seconds;
static double seconds_to_ticks;
static int hittimer;
static bool initialized=false;
static Mutex initlock("Time initialization lock");

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
Time::initialize()
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

    if(timer_32bit){
	    unsigned high=0;
	    unsigned low=*iotimer_addr32;
	    orig_timer=((long long)(high&(~TOPBIT))<<32|(long long)low);
    } else {
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
       for(;;) {
	    unsigned high = *iotimer_addr;
	    unsigned low = *(iotimer_addr + 1);
	    if (high == *iotimer_addr) {
		orig_timer=((long long)high<<32|(long long)low);
		break;
	    }
       }
#else
	orig_timer=*iotimer_addr-orig_timer;
#endif
    }

    iotimer_high=(*iotimer_addr32)&TOPBIT;

    if(timer_32bit){
	// Set up sigalrm handler...
	struct sigaction action;
	action.sa_flags=0;
	sigemptyset(&action.sa_mask);

	action.sa_handler=(SIG_PF)handle_alrm;
	if(sigaction(SIGALRM, &action, NULL) == -1)
	    throw ThreadError(std::string("sigaction failed")
			      +strerror(errno));

	int ticks=overflow/8;
	struct itimerval dt;
	dt.it_interval.tv_sec=ticks;
	dt.it_interval.tv_usec=0;
	dt.it_value.tv_sec=0;
	dt.it_value.tv_usec=1;
	struct itimerval old;
	if(setitimer(ITIMER_REAL, &dt, &old) != 0)
	    throw ThreadError(std::string("setitimer failed")
			      +strerror(errno));
	while(!hittimer)
	    sigsuspend(0);
    }
    initialized=true;
    initlock.unlock();
}

Time::SysClock
Time::currentTicks()
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
	for(;;) {
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
Time::currentSeconds()
{
    return Time::currentTicks()*ticks_to_seconds;
}

double
Time::secondsPerTick()
{
    if(!initialized)
	initialize();
    return ticks_to_seconds;
}

double
Time::ticksPerSecond()
{
    if(!initialized)
	initialize();
    return seconds_to_ticks;
}

void
Time::waitUntil(double seconds)
{
    waitFor(seconds-currentSeconds());
}

void
Time::waitFor(double seconds)
{
    if(!initialized)
	initialize();
    if(seconds<=0)
	return;
    static long tps=0;
    if(tps==0)
	tps=CLK_TCK;
    long ticks=(long)(seconds*(double)tps);
    int oldstate=Thread::couldBlock("Timed wait");
    while (ticks != 0){
	ticks=sginap(ticks);
    }
    Thread::couldBlockDone(oldstate);
}

void
Time::waitUntil(SysClock time)
{
    waitFor(time-currentTicks());
}

void
Time::waitFor(SysClock time)
{
    if(!initialized)
	initialize();
    if(time<=0)
	return;
    static double tps=0;
    if(tps==0)
	tps=(double)CLK_TCK*ticks_to_seconds;
    int ticks=time*tps;
    int oldstate=Thread::couldBlock("Timed wait");
    while (ticks != 0){
	ticks=(int)sginap(ticks);
    }
    Thread::couldBlockDone(oldstate);
}

