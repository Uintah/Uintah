#include <Uintah/Grid/DensityBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

DensityBoundCond::DensityBoundCond(double& rho) : d_rho(rho)
{
}

DensityBoundCond::DensityBoundCond(ProblemSpecP& ps)
{
  ps->require("density",d_rho);
}

DensityBoundCond::~DensityBoundCond()
{
}

double DensityBoundCond::getRho() const
{
  return d_rho;
}

std::string DensityBoundCond::getType() const
{
  return "Density";
}
