#include <Uintah/Grid/VelocityBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

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



