
#ifndef SCI_THREAD_TIME_H
#define SCI_THREAD_TIME_H

/**************************************
 
CLASS
   Time
   
KEYWORDS
   Time
   
DESCRIPTION
 
PATTERNS


WARNING
   
****************************************/

class RigorousTest;

class Time {
    Time();
    static void initialize();
public:
    typedef unsigned long long SysClock;

    //////////
    // Return the current system time, in terms of clock ticks.  Time
    // zero is at some arbitrary point in the past
    static SysClock currentTicks();
    
    //////////
    // Return the current system time, in terms of seconds.  This is
    // slower than currentTicks().  Time zero is at some arbitrary
    // point in the past.
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

    //////////
    // Testing interface
    static void test_rigorous(RigorousTest* __test);
};

#endif
