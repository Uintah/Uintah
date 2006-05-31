#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

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

const Point& ForceBC::getLowerRange() const
{
  return d_lowerRange;
}

const Point& ForceBC::getUpperRange() const
{
  return d_upperRange;
}

std::string ForceBC::getType() const
{
  return "Force";
}

