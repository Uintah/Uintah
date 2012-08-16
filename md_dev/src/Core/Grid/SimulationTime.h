/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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
  double max_wall_time;

  // Explicit number of timesteps to run.  Simulation runs either this
  // number of time steps, or to maxTime, which ever comes first.
  // if "max_Timestep" is not specified in the .ups file, then
  // max_Timestep == INT_MAX.  
  int    maxTimestep; 

  // Clamp the length of the timestep to the next
  // output or checkpoint if it will go over
  bool timestep_clamping;
  bool end_on_max_time;
private:
  SimulationTime(const SimulationTime&);
  SimulationTime& operator=(const SimulationTime&);
  
};

} // End namespace Uintah

#endif
