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

#include <Core/share/share.h>

class SCICORESHARE Timer {
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

class SCICORESHARE CPUTimer : public Timer {
    virtual double get_time();
public:
    virtual ~CPUTimer();
};

class SCICORESHARE WallClockTimer : public Timer {
    virtual double get_time();
public:
    WallClockTimer();
    virtual ~WallClockTimer();
};

class SCICORESHARE TimeThrottle : public WallClockTimer {
public:
    TimeThrottle();
    virtual ~TimeThrottle();
    void wait_for_time(double time);
};

#endif /* SCI_Containers_Timer_h */
