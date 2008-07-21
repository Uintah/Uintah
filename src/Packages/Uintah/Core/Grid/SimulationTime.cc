
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Thread/Thread.h>

#include <sci_values.h>
#include <string>
#include <iostream>

using namespace Uintah;

SimulationTime::SimulationTime(const ProblemSpecP& params)
{
  delt_factor = 1.0;
  override_restart_delt = 0.0;
  
  ProblemSpecP time_ps = params->findBlock("Time");
  time_ps->require("maxTime", maxTime);
  time_ps->require("initTime", initTime);
  time_ps->require("delt_min", delt_min);
  time_ps->require("delt_max", delt_max);
  time_ps->require("timestep_multiplier", delt_factor);
  if(!time_ps->get("delt_init", max_initial_delt)
     && !time_ps->get("max_initial_delt", max_initial_delt))
    max_initial_delt = DBL_MAX;
  if(!time_ps->get("initial_delt_range", initial_delt_range))
    initial_delt_range = 0;
  if(!time_ps->get("max_delt_increase", max_delt_increase))
    max_delt_increase=1.e99;
  if(!time_ps->get("max_wall_time",max_wall_time))
    max_wall_time=0;

  {
    // max_iterations is deprecated now... verify that it isn't used....
    int max_iterations = 0;
    if( time_ps->get( "max_iterations", max_iterations ).get_rep() != NULL ) {
      std::cerr << "\n";
      std::cerr << "The 'max_iterations' flag (in the .ups file) is deprecated.  Please use the 'max_Timesteps' flag instead..\n";
      std::cerr << "\n";
      SCIRun::Thread::exitAll(1);      
    }
  }

  // use INT_MAX -1, for some reason SGI optimizer doesn't like INT_MAX
  // in the SimulationController while loop
  maxTimestep = INT_MAX-1;
  time_ps->get( "max_Timesteps", maxTimestep );
  time_ps->get( "override_restart_delt", override_restart_delt);

  if (!time_ps->get("clamp_timesteps_to_output", timestep_clamping))
    timestep_clamping = false;

  if (!time_ps->get("end_on_max_time_exactly", end_on_max_time))
    end_on_max_time = false;

  if( maxTimestep < 1 )
    {
      std::cerr << "Negative maxTimesteps is not allowed.\n";
      std::cerr << "resetting to INT_MAX time steps\n";
      maxTimestep = INT_MAX-1;
    }
}

void SimulationTime::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP time_ps = params->findBlock("Time");
  time_ps->require("delt_min", delt_min);
  time_ps->require("delt_max", delt_max);
  time_ps->require("timestep_multiplier", delt_factor);

  if(!time_ps->get("delt_init", max_initial_delt)
     && !time_ps->get("max_initial_delt", max_initial_delt))
    max_initial_delt = DBL_MAX;
  if(!time_ps->get("initial_delt_range", initial_delt_range))
    initial_delt_range = 0;
  if(!time_ps->get("max_delt_increase", max_delt_increase))
    max_delt_increase=1.e99;
  
  time_ps->get( "override_restart_delt", override_restart_delt);

}
