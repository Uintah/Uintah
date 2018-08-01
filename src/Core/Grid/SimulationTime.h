/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CORE_GRID_SIMULATIONTIME_H
#define CORE_GRID_SIMULATIONTIME_H

#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************
      
  CLASS

    SimulationTime
      
      
  GENERAL INFORMATION
      
    SimulationTime.h
      
    Steven G. Parker
    Department of Computer Science
    University of Utah
      
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
  KEYWORDS

    SimulationTime
      

  DESCRIPTION

     
****************************************/
    
class SimulationTime {

public:

  SimulationTime(const ProblemSpecP& params);

  void problemSetup(const ProblemSpecP& params);

  // The simulation runs to either the maximum number of time steps
  // (maxTimestep) or the maximum simulation time (maxTime), which
  // ever comes first. If the "max_Timestep" is not specified in the .ups
  // file, then it is set to zero.
  
  int    m_max_time_steps{0};         // Maximum number of time steps to run.
  double m_max_time{0};               // Maximum simulation time
  double m_init_time{0};              // Initial simulation time
  
  double m_max_initial_delt{0};       // Maximum initial delta T
  double m_initial_delt_range{0};     // Simulation time range for the initial delta T
  double m_delt_min{0};               // Minimum delta T
  double m_delt_max{1};               // Maximum delta T
  double m_delt_factor{1};            // Factor for increasing delta T
  double m_max_delt_increase{0};      // Maximum delta T increase.
  double m_override_restart_delt{0};  // Override the restart delta T value

  double m_max_wall_time{0};          // Maximum wall time.
  
  bool m_clamp_time_to_output{false}; // Clamp the time to the next output or checkpoint
  bool m_end_at_max_time{false};      // End the simulation at exactly this time.
  
private:
  
  // eliminate copy, assignment and move
  SimulationTime( const SimulationTime & )            = delete;
  SimulationTime& operator=( const SimulationTime & ) = delete;
  SimulationTime( SimulationTime && )                 = delete;
  SimulationTime& operator=( SimulationTime && )      = delete;

}; // CORE_GRID_SIMULATIONTIME_H

} // End namespace Uintah

#endif
