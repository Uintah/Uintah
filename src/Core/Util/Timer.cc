/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <climits>
#include <iostream>
#include <Core/Thread/Thread.h>

#include <time.h>
using namespace std;
using namespace SCIRun;


using std::cerr;
using std::endl;

#if defined(CLK_TCK)
extern "C" long _sysconf(int);
#define CLOCK_INTERVAL CLK_TCK
#elif defined(CLOCKS_PER_SEC)
#define CLOCK_INTERVAL CLOCKS_PER_SEC
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
#if SCI_ASSERTION_LEVEL >=1  
  if(state != Stopped) {
    cerr << "Warning: Timer destroyed while it was running" << endl;
  }
#endif
}

void Timer::start()
{
  if (state == Stopped) {
    start_time = get_time();
    state = Running;
  } else {
    cerr << "Warning: Timer started while it was already running" << endl;
  }
}

void Timer::stop()
{
  if (state == Stopped) {
    cerr << "Warning: Timer stopped while it was already stopped" << endl;
  } else {
    state = Stopped;
    double t = get_time();
    total_time += t - start_time;
  }
}

void Timer::add(double t) {
    start_time -= t;
}

void Timer::clear()
{
  if (state == Stopped) {
    total_time = 0;
  } else {
    cerr << "Warning: Timer cleared while it was running" << endl;
    total_time = 0;
    start_time = get_time();
  }
}

double Timer::time()
{
  if (state == Running) {
    double t = get_time();
    return t - start_time + total_time;
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
  double cpu_time = double(buffer.tms_utime + buffer.tms_stime) / double(CLOCK_INTERVAL);
  return cpu_time;
}

WallClockTimer::WallClockTimer()
{
#ifdef __sgiasdf
  if(!checked_cycle_counter) {

    if(have_cycle_counter) {
    } else {
#endif
  if (ci == 0) {
    ci = 1. / double(CLOCK_INTERVAL);
  }
#ifdef __sgiasdf
}
#endif
}

WallClockTimer::~WallClockTimer() {}

double WallClockTimer::get_time()
{
#if 0
  struct timeval tp;
  if(gettimeofday(&tp) != 0) {
    cerr << "Time request failed!\n";
  }
  double time=double(tp.tv_sec)+double(tp.tv_usec)/1000000.;
#endif
  struct tms buffer;
  double time = double(times(&buffer)) * ci;
  return time;
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
  if (endtime == 0) {
    return;
  }
  double time_now = time();
  double delta = endtime - time_now;
  if (delta <= 0) {
    return;
  }
#ifdef __sgi
  int nticks=delta*CLOCK_INTERVAL;
  if(nticks<1) {
    return;
  }
  if(delta > 10) {
    cerr << "WARNING: delta=" << delta << endl;
  }
  sginap(nticks);
#else
  timespec delay, remaining;
  remaining.tv_sec = SCIRun::Floor(delta);
  remaining.tv_nsec = SCIRun::Floor((delta - SCIRun::Floor(delta)) * 1000000000);
  do {
    delay = remaining;
  } while (nanosleep(&delay, &remaining) != 0);
#endif
}

MultiThreadedTimer::MultiThreadedTimer(int numberOfThreads): timerLock("timerLock")  {
  //create by passing the number of threads.  You can determine this with:
  //int numThreads = Uintah::Parallel::getNumThreads();
  //if (numThreads == -1) {
  //  numThreads = 1;
 // }

  //To make sense of times, it helps to have the number of patches >= the number of threads.
  this->numberOfThreads = numberOfThreads;
  timers = new WallClockTimer[this->numberOfThreads];
  accumulatedTimes = new double[this->numberOfThreads];
  for (int i = 0; i < this->numberOfThreads; i++) {
    accumulatedTimes[i] = 0;
  }
}

MultiThreadedTimer::~MultiThreadedTimer() {
  delete[] timers;
  delete[] accumulatedTimes;
}
void MultiThreadedTimer::start() {

  int threadID = SCIRun::Thread::self()->myid();
  timers[threadID].start();
}

void MultiThreadedTimer::stop(){
  timers[SCIRun::Thread::self()->myid()].stop();
}
void MultiThreadedTimer::clear(){
  int threadID = SCIRun::Thread::self()->myid();
  accumulatedTimes[threadID] += time();
  timers[threadID].clear();
}

void MultiThreadedTimer::clearAll(){
  //To get accurate results, only call when you know one
  //thread will be accessing methods of this class.
  for (int i = 0; i < numberOfThreads; i++) {
    accumulatedTimes[i] += timers[i].time();
    timers[i].clear();
  }
}


double MultiThreadedTimer::time(){
  //timerLock.writeLock();
  return timers[SCIRun::Thread::self()->myid()].time();
  //timerLock.writeUnlock();
}

double MultiThreadedTimer::getElapsedSeconds(){
  int threadID = SCIRun::Thread::self()->myid();
  return timers[threadID].time();
}

double MultiThreadedTimer::getElapsedAccumulatedSeconds(){
  int threadID = SCIRun::Thread::self()->myid();
  return accumulatedTimes[threadID] + timers[threadID].time();
}

double MultiThreadedTimer::getAllThreadsElapsedSeconds(){
  //To get accurate results, only call when all other
  //threads have finished their times.  Also read the last
  //entry.
  //timerLock.readLock();

  double total = 0.0;
  for (int i = 0; i < numberOfThreads; i++) {
    total += timers[i].time();
  }
  //timerLock.readUnlock();
  return total;


}

double MultiThreadedTimer::get_time(){
  return 0;
}

