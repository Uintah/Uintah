#ifndef UINTAH_HOMEBREW_SimulationTime_H
#define UINTAH_HOMEBREW_SimulationTime_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

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
      
    Copyright (C) 2000 SCI Group
      
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
  double maxTime;
  double initTime;
  double max_initial_delt;
  double initial_delt_range;
  double delt_min;
  double delt_max;
  double delt_factor;
  double max_delt_increase;
  double override_restart_delt; 

  // Explicit number of timesteps to run.  Simulation runs either this
  // number of time steps, or to maxTime, which ever comes first.
  // if "max_iterations" is not specified in the .ups file, then
  // max_iterations == MAXINT.  
  // The difference between max_iterations and num_time_steps is 
  // that on a restart, max_iterations will still go max_iterations,
  // but maxTimestep will go until that time step.
  int    max_iterations;
  int    maxTimestep; 

  // Clamp the length of the timestep to the next
  // output or checkpoint if it will go over
  bool timestep_clamping;
private:
  SimulationTime(const SimulationTime&);
  SimulationTime& operator=(const SimulationTime&);
  
};

} // End namespace Uintah

#endif
