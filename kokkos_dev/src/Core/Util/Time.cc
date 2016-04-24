/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
 *  Time.cc: Generic UNIX implementation of the Time class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 */

#include <Core/Util/Time.h>

#include <stdexcept>
#include <cstdio>
#include <sys/time.h>
#include <cstring>
#include <cerrno>

#ifdef __linux
#include <time.h>
#endif

static bool initialized = false;
static struct timeval start_time;

using namespace Uintah;

void Time::initialize()
{
  initialized = true;
  if (gettimeofday(&start_time, 0) != 0)
    throw std::runtime_error("gettimeofday failed: ");
}

double Time::secondsPerTick()
{
  return 1.e-6;
}

double Time::currentSeconds()
{
  if (!initialized)
    initialize();
  struct timeval now_time;
  if (gettimeofday(&now_time, 0) != 0)
    throw std::runtime_error("gettimeofday failed: ");

  return (now_time.tv_sec - start_time.tv_sec) + (now_time.tv_usec - start_time.tv_usec) * 1.e-6;
}

Time::SysClock Time::currentTicks()
{
  if (!initialized)
    initialize();
  struct timeval now_time;
  if (gettimeofday(&now_time, 0) != 0)
    throw std::runtime_error("gettimeofday failed: ");

  return (now_time.tv_sec - start_time.tv_sec) * 1000000 + (now_time.tv_usec - start_time.tv_usec);
}

double Time::ticksPerSecond()
{
  return 1000000;
}

void Time::waitUntil(double seconds)
{
  waitFor(seconds - currentSeconds());
}

void Time::waitFor(double seconds)
{
  if (!initialized)
    initialize();

  if (seconds <= 0)
    return;

  struct timespec ts;
  ts.tv_sec = (int)seconds;
  ts.tv_nsec = (int)(1.e9 * (seconds - ts.tv_sec));

  nanosleep(&ts, &ts);
}

void Time::waitUntil(SysClock time)
{
  waitFor(time - currentTicks());
}

void Time::waitFor(SysClock time)
{
  if (!initialized)
    initialize();
  if (time <= 0)
    return;
  struct timespec ts;
  ts.tv_sec = (int)(time * 1.e-6);
  ts.tv_nsec = (int)(1.e9 * (time * 1.e-6 - ts.tv_sec));

  nanosleep(&ts, &ts);
}
