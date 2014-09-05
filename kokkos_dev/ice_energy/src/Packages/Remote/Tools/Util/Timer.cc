//////////////////////////////////////////////////////////////////////
// Timer.cpp - A wall clock timer.
// By Dave McAllister, 1998.

#include <Packages/Remote/Tools/Util/Timer.h>

#include <sys/times.h>
#include <limits.h>
#include <unistd.h>
static double clock_interval = 1./double(CLK_TCK);

namespace Remote {
// Create the timer. It is stopped.
Timer::Timer()
{
	Going = false;
	StartTime = GetCurTime();
	ElapsedTime = 0;
}

// Start or re-start the timer.
double Timer::Start()
{
	Going = true;
	StartTime = GetCurTime();

	return ElapsedTime;
}

// Stop the timer and set ElapsedTime to be the total time it's run so far.
double Timer::Stop()
{
	Going = false;
	double CurTime = GetCurTime();
	ElapsedTime += CurTime - StartTime;
	StartTime = CurTime;

	return ElapsedTime;
}

// Return elapsed time on the timer.
// If the clock is still going, add in how long it's been going
// since the last time it was started.
// If it's not going, it's just ElapsedTime.
double Timer::Read()
{
	if(Going)
		return ElapsedTime + (GetCurTime() - StartTime);
	else
		return ElapsedTime;
}

// Reset the elapsed time to 0.
// Doesn't start or stop the clock. This is like Dad's old
// silver stopwatch. Return the elapsed time *before* it was reset.
double Timer::Reset()
{
	double CurTime = GetCurTime();
	double El = ElapsedTime + (CurTime - StartTime);
	StartTime = CurTime;
	ElapsedTime = 0;
	
	return El;
}

double GetCurTime()
{
    struct tms buffer;
    double dtime=double(times(&buffer)) * clock_interval;
    return dtime;
}
} // End namespace Remote

