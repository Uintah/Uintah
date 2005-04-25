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
 *  Time_win32.cc: Generic unix implementation of the Time class
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
#include <windows.h>
#include <Mmsystem.h> // windows time functions
#include <errno.h>

static bool initialized=false;
static DWORD start_time; // measured in milliseconds

using namespace SCIRun;

void
Time::initialize()
{
  initialized=true;
  timeBeginPeriod(1); // give the timer millisecond accuracy
  start_time = timeGetTime();
}

double
Time::secondsPerTick()
{
  return 1.e-3;
}

double
Time::currentSeconds()
{
  if(!initialized)
    initialize();
  DWORD now_time = timeGetTime();

  return ((double)(now_time - start_time))*1.e-3; 
}

Time::SysClock
Time::currentTicks()
{ 
  if(!initialized)
    initialize();
  DWORD now_time = timeGetTime();

  return now_time - start_time;
}

double
Time::ticksPerSecond()
{
    return 1000;
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

  int oldstate=Thread::couldBlock("Timed wait");
  Sleep(seconds*1000); // Sleep is in milliseconds
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
  int oldstate=Thread::couldBlock("Timed wait");
  Sleep(time);
  Thread::couldBlockDone(oldstate);
}

