
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


#include <Thread/Time.h>
#include <Thread/Thread.h>
#include <stdio.h>
#include <sys/time.h>

static bool initialized=false;
static struct timeval start_time;

void Time::initialize()
{
    initialized=true;
    if(gettimeofday(&start_time, 0) != 0){
	perror("getimeofday");
	Thread::niceAbort();
    }
}

double Time::secondsPerTick()
{
    return 1.e-6;
}

double Time::currentSeconds()
{
    if(!initialized)
	initialize();
    struct timeval now_time;
    if(gettimeofday(&now_time, 0) != 0){
	perror("gettimeofday");
	Thread::niceAbort();
    }
    return (now_time.tv_sec-start_time.tv_sec)+(now_time.tv_usec-start_time.tv_usec)*1.e-6;
}

Time::SysClock Time::currentTicks()
{ 
    if(!initialized)
	initialize();
    struct timeval now_time;
    if(gettimeofday(&now_time, 0) != 0){
	perror("gettimeofday");
	Thread::niceAbort();
    }
    return (now_time.tv_sec-start_time.tv_sec)*100000+(now_time.tv_usec-start_time.tv_usec);
}

double Time::ticksPerSecond()
{
    return 1000000;
}

void Time::waitUntil(double seconds)
{
    waitFor(seconds-currentSeconds());
}

void Time::waitFor(double seconds)
{
    if(!initialized)
	initialize();
    if(seconds<=0)
	return;
    struct timespec ts;
    ts.tv_sec=(int)seconds;
    ts.tv_nsec=(int)(1.e9*(seconds-ts.tv_sec));
    while (nanosleep(&ts, &ts) == 0) /* Nothing */ ;
}

void Time::waitUntil(SysClock time)
{
    waitFor(time-currentTicks());
}

void Time::waitFor(SysClock time)
{
    if(!initialized)
	initialize();
    if(time<=0)
	return;
    struct timespec ts;
    ts.tv_sec=(int)(time*1.e-6);
    ts.tv_nsec=(int)(1.e9*(time*1.e-6-ts.tv_sec));
    while (nanosleep(&ts, &ts) == 0) /* Nothing */ ;
}

//
// $Log$
// Revision 1.1  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

