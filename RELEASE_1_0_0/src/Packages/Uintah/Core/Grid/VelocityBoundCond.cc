#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

VelocityBoundCond::VelocityBoundCond(ProblemSpecP& ps,const std::string& kind)
  : BoundCond<Vector>(kind)
{
  d_type = "Velocity";
  ps->require("value",d_vel);
}

VelocityBoundCond::~VelocityBoundCond()
{
}

Vector VelocityBoundCond::getValue() const
{
  return d_vel;
}



