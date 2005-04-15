#include <Uintah/Grid/KinematicBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

KinematicBoundCond::KinematicBoundCond(Vector& v)
{
  d_vel = v;
}

KinematicBoundCond::KinematicBoundCond(ProblemSpecP& ps)
{
  ps->require("velocity",d_vel);
}

KinematicBoundCond::~KinematicBoundCond()
{
}

Vector KinematicBoundCond::getVelocity() const
{
  return d_vel;
}

std::string KinematicBoundCond::getType() const
{
  return "Kinematic";
}

