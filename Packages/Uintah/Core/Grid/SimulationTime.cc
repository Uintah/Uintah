
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <values.h>

using namespace Uintah;

SimulationTime::SimulationTime(const ProblemSpecP& params)
{
  delt_factor = 1.0;

  ProblemSpecP time_ps = params->findBlock("Time");
  time_ps->require("maxTime", maxTime);
  time_ps->require("initTime", initTime);
  time_ps->require("delt_min", delt_min);
  time_ps->require("delt_max", delt_max);
  time_ps->require("timestep_multiplier", delt_factor);

  string max_iters;
  if( time_ps->getOptional( "max_iterations", max_iters ) )
    {
      num_time_steps = atoi( max_iters.c_str() );
      if( num_time_steps < 1 )
	{
	  cerr << "Negative number of time steps is not allowed.\n";
	  cerr << "reseting to MAXINT time steps\n";
	  num_time_steps = MAXINT;
	}
    }
  else
    {
      num_time_steps = MAXINT;
    }
}

