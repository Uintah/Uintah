//////////////////////////////////////////////////////////////////////
// Timer.h - A wall clock timer.
// By Dave McAllister, 1998.

// Simulates my dad's old stop watch. It had a stop / start switch and a
// reset button that were independent. First press Reset(), then press
// Start(). Then press Stop() and then Read() it. Also, you can Read() it
// while it's going, and you can Start() and Stop() it multiple times
// without Reset()ing it, and the time will accumulate. Also, you can
// Reset() it while it is going and it will reset to 0 and go from there.
// Note that although the time is stored as a double, the clock is not very
// accurate. It has maybe 1 ms granularity or worse.

#ifndef _timer_h
#define _timer_h

namespace Remote {
class Timer
{
	double StartTime;   // The time the clock was most-recently started.
	double ElapsedTime; // Total of the time-spans that it was started then stopped.
	bool Going;         // Is the clock going now?

public:
	// Create a new stopwatch. It is stopped.
	Timer();

	// Start the timer and return the total time it's run so far.
	double Start();

	// Stop the timer and return the total time it's run so far.
	double Stop();

	// Returns the time in seconds and fractions of a second.
	double Read();

	// Reset the elapsed time to 0. Doesn't start or stop the
	// clock. This is like Dad's old silver stopwatch. Return the
	// elapsed time *before* it was reset.
	double Reset();

	// Returns seconds since the epoch.
	// Warning: On unix, this returns 0 if called before main(),
	// such as in a constructor.
	friend double GetCurTime();
};

} // End namespace Remote


#endif
