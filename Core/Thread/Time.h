
/*
 *  Time: Utility class for dealing with time
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Time_h
#define Core_Thread_Time_h

#include <Core/share/share.h>

namespace SCIRun {
/**************************************

 CLASS
 Time

 KEYWORDS
 Thread, Time

 DESCRIPTION
 Utility class to manage Time.  This class is implemented using
 high precision counters on the SGI, and standard unix system calls
 on other machines.

****************************************/
class SCICORESHARE Time {
public:
  typedef unsigned long long SysClock;
	    
  //////////
  // Return the current system time, in terms of clock ticks.
  // Time zero is at some arbitrary point in the past.
  static SysClock currentTicks();
	    
  //////////
  // Return the current system time, in terms of seconds.
  // This is slower than currentTicks().  Time zero is at
  // some arbitrary point in the past.
  static double currentSeconds();
	    
  //////////
  // Return the conversion from seconds to ticks.
  static double ticksPerSecond();
	    
  //////////
  // Return the conversion from ticks to seconds.
  static double secondsPerTick();
	    
  //////////
  // Wait until the specified time in clock ticks.
  static void waitUntil(SysClock ticks);
	    
  //////////
  // Wait until the specified time in seconds.
  static void waitUntil(double seconds);
	    
  //////////
  // Wait for the specified time in clock ticks
  static void waitFor(SysClock ticks);
	    
  //////////
  // Wait for the specified time in seconds
  static void waitFor(double seconds);

private:
  Time();
  static void initialize();
};
} // End namespace SCIRun

#endif


