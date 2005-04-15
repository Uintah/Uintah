
#include <Uintah/Grid/SimulationTime.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <string>

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
}

//
// $Log$
// Revision 1.3  2000/06/08 20:59:19  jas
// Added timestep multiplier (fudge factor).
//
// Revision 1.2  2000/04/26 06:48:56  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/13 06:51:02  sparker
// More implementation to get this to work
//
//
