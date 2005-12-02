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

#ifndef __Timer_h
#define __Timer_h

namespace SemotusVisum {

/**
 * Interface to portable timer utility classes
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Timer {
  /// Total time elapsed
  double total_time;
  /// Start time
  double start_time;

  /// Current timer state
  enum State {
    Stopped,
    Running
  };

  /// Current timer state
  State state;

  /// Get the current time
  virtual double get_time()=0;
public:
  
  /// Constructor
  Timer();

  /// Destructor
  virtual ~Timer();

  /// Starts the timer
  void start();

  /// Stops the timer
  void stop();

  /// Clears the timer
  void clear();

  /**
   *  Returns the current elapsed time, in seconds
   *
   * @return Current elapsed time
   */
  double time();

  
  /**
   * Adds time to the timer
   *
   * @param t     Time to add
   */
  void add(double t);
};


/**
 * Returns the amount of CPU time used (user and system)
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class CPUTimer : public Timer {
  /// Get the current time
  virtual double get_time();
public:
  /// Destructor
  virtual ~CPUTimer();
};

/**
 * Returns the amount of wall clock time used.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class WallClockTimer : public Timer {
  /// Get the current time
  virtual double get_time();
public:
  /// Constructor
  WallClockTimer();
  /// Destructor
  virtual ~WallClockTimer();
};

/**
 * Returns the precise amount of time used.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class PreciseTimer : public Timer {
  /// Get the current time
  virtual double get_time();
public:
  /// Constructor
  PreciseTimer();
  /// Destructor
  virtual ~PreciseTimer();
};


/**
 * Allows us to wait for a particular time
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class TimeThrottle : public WallClockTimer {
public:
  /// Constructor
  TimeThrottle();

  /// Destructor
  virtual ~TimeThrottle();
  
  /**
   * Waits until the current time matches that given in 'time'
   *
   * @param time  Absolute time - this will typically be large!
   */
  void wait_for_time(double time);
};

}

#endif /* __Timer_h */
