#include <Uintah/Grid/DensityBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

DensityBoundCond::DensityBoundCond(ProblemSpecP& ps, std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "Density";
  ps->require("value",d_rho);
}

DensityBoundCond::~DensityBoundCond()
{
}

double DensityBoundCond::getValue() const
{
  return d_rho;
}

