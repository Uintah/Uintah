#include <Uintah/Components/MPM/PhysicalBC/ForceBC.h>

#include <Uintah/Interface/ProblemSpec.h>

using namespace Uintah::MPM;

ForceBC::ForceBC(ProblemSpecP& ps)
{
  ps->require("lower",d_lowerRange);
  ps->require("upper",d_upperRange);
  ps->require("force_density",d_forceDensity);
}

const Vector& ForceBC::getForceDensity() const
{
  return d_forceDensity;
}

const Vector& ForceBC::getLowerRange() const
{
  return d_lowerRange;
}

const Vector& ForceBC::getUpperRange() const
{
  return d_upperRange;
}

std::string ForceBC::getType() const
{
  return "force";
}

// $Log$
// Revision 1.1  2000/08/07 00:42:51  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
