
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#if defined(_WIN32) || defined(__APPLE__)
#include <float.h>
#define MAXDOUBLE DBL_MAX
#define MAXINT INT_MAX
#else
#include <values.h>
#endif
#include <string>
#include <iostream>

using namespace Uintah;
using namespace std;

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
    max_initial_delt = MAXDOUBLE;
  if(!time_ps->get("initial_delt_range", initial_delt_range))
    initial_delt_range = 0;
  if(!time_ps->get("max_delt_increase", max_delt_increase))
    max_delt_increase=1.e99;

  num_time_steps = MAXINT;
  time_ps->get( "max_iterations", num_time_steps );
  time_ps->get( "override_restart_delt", override_restart_delt);

  if (!time_ps->get("clamp_timesteps_to_output", timestep_clamping))
    timestep_clamping = false;
  

  if( num_time_steps < 1 )
    {
      cerr << "Negative number of time steps is not allowed.\n";
      cerr << "reseting to MAXINT time steps\n";
      num_time_steps = MAXINT;
    }
}

