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


#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadError.h>
#include <stdio.h>
#include <sys/time.h>
#include <errno.h>
#ifdef __linux
#include <time.h>
#endif

static bool initialized=false;
static struct timeval start_time;

using namespace SCIRun;

void
Time::initialize()
{
    initialized=true;
    if(gettimeofday(&start_time, 0) != 0)
	throw ThreadError(std::string("gettimeofday failed: ")
			  +strerror(errno));
}

double
Time::secondsPerTick()
{
    return 1.e-6;
}

double
Time::currentSeconds()
{
    if(!initialized)
	initialize();
    struct timeval now_time;
    if(gettimeofday(&now_time, 0) != 0)
	throw ThreadError(std::string("gettimeofday failed: ")
			  +strerror(errno));

    return (now_time.tv_sec-start_time.tv_sec)+(now_time.tv_usec-start_time.tv_usec)*1.e-6;
}

Time::SysClock
Time::currentTicks()
{ 
    if(!initialized)
	initialize();
    struct timeval now_time;
    if(gettimeofday(&now_time, 0) != 0)
	throw ThreadError(std::string("gettimeofday failed: ")
			  +strerror(errno));

    return (now_time.tv_sec-start_time.tv_sec)*1000000+(now_time.tv_usec-start_time.tv_usec);
}

double
Time::ticksPerSecond()
{
    return 1000000;
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

  struct timespec ts;
  ts.tv_sec=(int)seconds;
  ts.tv_nsec=(int)(1.e9*(seconds-ts.tv_sec));

  int oldstate=Thread::couldBlock("Timed wait");
  nanosleep(&ts, &ts);
  //  while (nanosleep(&ts, &ts) == 0) /* Nothing */ ;
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
    struct timespec ts;
    ts.tv_sec=(int)(time*1.e-6);
    ts.tv_nsec=(int)(1.e9*(time*1.e-6-ts.tv_sec));
    int oldstate=Thread::couldBlock("Timed wait");
    nanosleep(&ts, &ts);
    //while (nanosleep(&ts, &ts) == 0) /* Nothing */ ;
    Thread::couldBlockDone(oldstate);
}

