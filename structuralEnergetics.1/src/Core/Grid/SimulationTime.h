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

#ifndef UINTAH_HOMEBREW_SimulationTime_H
#define UINTAH_HOMEBREW_SimulationTime_H

#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************
      
  CLASS
    SimulationTime
      
    Short Description...
      
  GENERAL INFORMATION
      
    SimulationTime.h
      
    Steven G. Parker
    Department of Computer Science
    University of Utah
      
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
  KEYWORDS
    SimulationTime
      
  DESCRIPTION
    Long description...
      
  WARNING
     
****************************************/
    
class SimulationTime {
public:
  SimulationTime(const ProblemSpecP& params);

  void problemSetup(const ProblemSpecP& params);

  // The simulation runs to either the maximum number of time steps
  // (maxTimestep) or the maximum simulation time (maxTime), which
  // ever comes first. If the "max_Timestep" is not specified in the .ups
  // file, then it is set to INT_MAX-1.
  
  int    maxTimestep;             // Maximum number of time steps to run.
  double maxTime;                 // Maximum simulation time
  double initTime;                // Initial simulation time
  
  double max_initial_delt;        // Maximum initial delta T
  double initial_delt_range;      // Simulation time to use the initial delta T
  double delt_min;                // Minimum delta T
  double delt_max;                // Maximum delta T
  double delt_factor;             // Factor for increasing delta T
  double max_delt_increase;       // Maximum delta T increase.
  double override_restart_delt;   // Overirde the restart delta T value

  double max_wall_time;           // Maximum wall time.
  
  bool clamp_time_to_output; // Clamp the time to the next output or checkpoint
  bool end_at_max_time;      // End the simulation at exactly this time.
  
private:
  SimulationTime(const SimulationTime&);
  SimulationTime& operator=(const SimulationTime&);
  
};

} // End namespace Uintah

#endif
