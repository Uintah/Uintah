
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

#ifndef SCI_Containers_Timer_h
#define SCI_Containers_Timer_h 1

#include <share/share.h>

class SHARE Timer {
    double total_time;
    double start_time;
    enum State {
	Stopped,
	Running
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
    void add(double t);
};

class SHARE CPUTimer : public Timer {
    virtual double get_time();
public:
    virtual ~CPUTimer();
};

class SHARE WallClockTimer : public Timer {
    virtual double get_time();
public:
    WallClockTimer();
    virtual ~WallClockTimer();
};

class SHARE TimeThrottle : public WallClockTimer {
public:
    TimeThrottle();
    virtual ~TimeThrottle();
    void wait_for_time(double time);
};

#endif /* SCI_Containers_Timer_h */
