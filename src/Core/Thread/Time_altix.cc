
/*
 For more information, please see: http://software.sci.utah.edu
 
 The MIT License
 
 Copyright (c) 2005 Silicon Graphics Inc.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
 */



/*
 *  Time_altix.cc: Timer implementation using memory mapped mmtimer
 *                 For SGI Altix and Prism systems.
 *
 *  Written by:
 *   Author: Abe Stephens
 *   Date: July 2005
 *
 *   Implementation follows example provided by Brian Sumner (SGI)
 *
 *  Copyright (C) 2005 SGI
 */


#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadError.h>

#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cerrno>

#include <sn/mmtimer.h>

#include <string>
using namespace std;

static bool initialized = false;
using namespace SCIRun;

typedef Time::SysClock SysClock;

static double             mmtimer_scale;  // Seconds per tick.
static SysClock           mmtimer_base;   // Tick value at initialization.
static volatile SysClock *mmtimer_addr;   // Pointer to the memory mapped timer.

void Time::initialize() {

	// Only initialize once.
	if (initialized) 
		return;
	initialized = true;

	int fd, offset;
	unsigned long femtosecs_per_tick;
	char *m_page;

	// First open the mmtimer, then memory map it.	
	if ((fd = open( MMTIMER_FULLNAME, O_RDONLY, 0 )) < 0) {
		throw ThreadError( string("MMTIMER_FULLNAME failed: ") + strerror( errno ) );
	}
	
	// Check to see if we can memory map the timer.
	if (ioctl(fd, MMTIMER_MMAPAVAIL, 0) <= 0) {
		throw ThreadError( string("MMTIMER_MMAPAVAIL failed: ") + strerror( errno ) );
	}

	
	// Determine the offset in the mapped page.
	if ((offset = ioctl(fd, MMTIMER_GETOFFSET, 0)) < 0) {
		throw ThreadError( string("MMTIMER_GETOFFSET failed: ") + strerror( errno ) );
	}
	
	// Determine the timer resolution.
	if (ioctl(fd, MMTIMER_GETRES, &femtosecs_per_tick) < 0) {
		throw ThreadError( string("MMTIMER_GETRES failed: ") + strerror( errno ) );
	}
	
	// Determine seconds per tick (femto/tick) / (femto/sec)
	mmtimer_scale = (double)femtosecs_per_tick / 1.0e+15;
	
	// Map the timer.
	if ((m_page = (char *)mmap( 0, getpagesize(), PROT_READ, MAP_SHARED, fd, 0 )) == MAP_FAILED) {
		throw ThreadError( string("mmap of mmtimer failed: ") + strerror( errno ) );
	}
	
	// All done with the fd.
	close( fd );
	
	// Determine the timer address.
	mmtimer_addr  = (volatile SysClock *)(m_page + offset);
	
	// Initial time.
	mmtimer_base  = *mmtimer_addr;
}

double Time::secondsPerTick() {
  if (!initialized)
    initialize();
  
	return mmtimer_scale;
}

double Time::currentSeconds() {
	if(!initialized)
		initialize();
	
	return (double)((*mmtimer_addr) - mmtimer_base) * mmtimer_scale;
}

Time::SysClock Time::currentTicks() { 
	
	if(!initialized)
		initialize();
		
	return (Time::SysClock)((*mmtimer_addr) - mmtimer_base);
}

double Time::ticksPerSecond() {
	return 1.0 / secondsPerTick();
}

void Time::waitUntil(double seconds) {
	waitFor(seconds-currentSeconds());
}

void Time::waitFor(double seconds) {
	// Convert seconds to ticks.
	SysClock ticks = ticksPerSecond() * seconds;
	waitFor( ticks );
}

void Time::waitUntil(SysClock time) {
	waitFor(time-currentTicks());
}

void Time::waitFor(SysClock ticks) {
	if(!initialized)
		initialize();

	if(time <= 0)
		return;

	int oldstate=Thread::couldBlock("Timed wait");

	// Busy wait.
	SysClock end = (*mmtimer_addr) + ticks;
	while ((*mmtimer_addr) < end);

	Thread::couldBlockDone(oldstate);
}

