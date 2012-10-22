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
 *  Timer.h: Implementation of portable timer utility classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 */


#include <Core/Util/Timer.h>
#include <Core/Util/NotFinished.h>
#include <Core/Thread/Thread.h>
#include <Core/Math/MiscMath.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#else
#include <windows.h>
#endif
#include <climits>
#include <iostream>
using std::cerr;
using std::endl;


#include <time.h>
#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

#ifndef _CLOCK_T_DEFINED
#define _CLOCK_T_DEFINED
typedef long clock_t;
#endif

struct tms
{
    clock_t tms_utime;
    clock_t tms_stime;
    clock_t tms_cutime;
    clock_t tms_cstime;
};

#ifdef __cplusplus
}
#endif
#include <sys/timeb.h>
#endif

#if defined(CLK_TCK)
extern "C" long _sysconf(int);
#define CLOCK_INTERVAL CLK_TCK
#elif defined(CLOCKS_PER_SEC)
#define CLOCK_INTERVAL CLOCKS_PER_SEC
#else
#include <sys/param.h>
#define CLOCK_INTERVAL HZ
#endif

#ifdef _WIN32
clock_t times(struct tms* buffer)
{
	timeb curtime;
	ftime(&curtime);

	long ticks = (curtime.time*1000)+curtime.millitm;

	buffer->tms_utime = buffer->tms_cutime = ticks;
	buffer->tms_stime = buffer->tms_cstime = 0;

	return ticks;

	//cerr << "times() called" << endl;
}
#endif

static double ci=0;

Timer::Timer()
{
    state=Stopped;
    total_time=0;
}

Timer::~Timer()
{
#if SCI_ASSERTION_LEVEL >=1  
    if(state != Stopped){
	cerr << "Warning: Timer destroyed while it was running" << endl;
    }
#endif
}

void Timer::start()
{
    if(state == Stopped){
	start_time=get_time();
	state=Running;
    } else {
	cerr << "Warning: Timer started while it was already running" << endl;
    }
}

void Timer::stop()
{
    if(state == Stopped){
	cerr << "Warning: Timer stopped while it was already stopped" << endl;
    } else {
	state=Stopped;
	double t=get_time();
	total_time+=t-start_time;
    }
}

void Timer::add(double t) {
    start_time -= t;
}

void Timer::clear()
{
    if(state == Stopped){
	total_time=0;
    } else {
	cerr << "Warning: Timer cleared while it was running" << endl;
	total_time=0;
	start_time=get_time();
    }
}

double Timer::time()
{
    if(state == Running){
	double t=get_time();
	return t-start_time+total_time;
    } else {
	return total_time;
    }
}


double CPUTimer::get_time()
{
#if 0
    struct rusage cpu_usage;
    getrusage(RUSAGE_SELF, &cpu_usage);
    double cpu_time=
	 double(cpu_usage.ru_utime.tv_sec)
	+double(cpu_usage.ru_stime.tv_sec)
	+double(cpu_usage.ru_utime.tv_sec)/1000000.
	+double(cpu_usage.ru_stime.tv_sec)/1000000.;
#endif
    struct tms buffer;
    times(&buffer);
    double cpu_time=
	double(buffer.tms_utime+buffer.tms_stime)/double(CLOCK_INTERVAL);
    return cpu_time;
}

WallClockTimer::WallClockTimer()
{
#ifdef __sgiasdf
  if(!checked_cycle_counter){
    

    if(have_cycle_counter){
    } else {
#endif
      if(ci==0)
	ci=1./double(CLOCK_INTERVAL);
#ifdef __sgiasdf
    }
#endif
}

double WallClockTimer::get_time()
{
#if 0
    struct timeval tp;
    if(gettimeofday(&tp) != 0){
	cerr << "Time request failed!\n";
    }
    double time=double(tp.tv_sec)+double(tp.tv_usec)/1000000.;
#endif
    struct tms buffer;
    double time=double(times(&buffer))*ci;
    return time;
}

WallClockTimer::~WallClockTimer()
{
}

CPUTimer::~CPUTimer()
{
}

TimeThrottle::TimeThrottle()
{
}

TimeThrottle::~TimeThrottle()
{
}

void TimeThrottle::wait_for_time(double endtime)
{
    if(endtime==0)
	return;
    double time_now=time();
    double delta=endtime-time_now;
    if(delta <=0)
	return;
#ifdef __sgi
    int nticks=delta*CLOCK_INTERVAL;
    if(nticks<1)return;
    if(delta > 10){
	cerr << "WARNING: delta=" << delta << endl;
    }
    sginap(nticks);
#else
#ifdef _WIN32
    Sleep(delta*1e3); // windows Sleep is in ms
#else
    timespec delay, remaining;
    remaining.tv_sec = SCIRun::Floor(delta);
    remaining.tv_nsec = SCIRun::Floor((delta-SCIRun::Floor(delta))*1000000000);
    do {
      delay = remaining;
    } while (nanosleep(&delay,&remaining) != 0);
#endif
#endif
}
