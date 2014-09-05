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
 *  Timer.h: Implementation of portable timer utility classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Core/Util/Timer.h>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <limits.h>
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

#ifdef CLK_TCK
extern "C" long _sysconf(int);
#define CLOCK_INTERVAL CLK_TCK
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
    if(state != Stopped){
	cerr << "Warning: Timer destroyed while it was running" << endl;
    }
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
    //NOT_FINISHED("TimeThrottle::wait_for_time");
#endif
}
