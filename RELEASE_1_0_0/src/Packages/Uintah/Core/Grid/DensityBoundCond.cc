#include <Packages/Uintah/Core/Grid/DensityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

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

