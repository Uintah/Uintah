
#include <Uintah/Grid/SimulationTime.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <string>

using namespace Uintah::Grid;

SimulationTime::SimulationTime(const ProblemSpecP& params)
{
  ProblemSpecP time_ps = params->findBlock("Time");
  time_ps->require("maxTime", maxTime);
  time_ps->require("initTime", initTime);
  time_ps->require("delt_min", delt_min);
  time_ps->require("delt_max", delt_max);
}

//
// $Log$
// Revision 1.1  2000/04/13 06:51:02  sparker
// More implementation to get this to work
//
//
