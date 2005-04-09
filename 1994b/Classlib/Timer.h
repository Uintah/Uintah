
/*
 *  Timer.h: Interface to portable timer utility classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_Timer_h
#define SCI_Classlib_Timer_h 1

class Timer {
    double total_time;
    double start_time;
    enum State {
	Stopped,
	Running,
    };
    State state;
    virtual double get_time()=0;
public:
    Timer();
    virtual ~Timer();
    void start();
    void stop();
    void clear();
    double time();
};

class CPUTimer : public Timer {
    virtual double get_time();
public:
    virtual ~CPUTimer();
};

class WallClockTimer : public Timer {
    virtual double get_time();
public:
    WallClockTimer();
    virtual ~WallClockTimer();
};

#endif /* SCI_Classlib_Timer_h */
