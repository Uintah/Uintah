
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


#include <Classlib/Timer.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <limits.h>
#include <iostream.h>
#include <unistd.h>

#ifdef CLK_TCK
extern "C" long _sysconf(int);
#define CLOCK_INTERVAL CLK_TCK
#else
#include <sys/param.h>
#define CLOCK_INTERVAL HZ
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
    if(ci==0)
	ci=1./double(CLOCK_INTERVAL);
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
