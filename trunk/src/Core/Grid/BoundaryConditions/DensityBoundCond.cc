#include <Packages/Uintah/Core/Grid/BoundaryConditions/DensityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

DensityBoundCond::DensityBoundCond(ProblemSpecP& ps, std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "Density";
  ps->require("value",d_value);
  if (kind == "Dirichlet_perturbed")
    ps->require("constant",d_constant);
  else
    d_constant = 0.;
}

DensityBoundCond::~DensityBoundCond()
{
}

DensityBoundCond* DensityBoundCond::clone()
{
  return scinew DensityBoundCond(*this);
}

double DensityBoundCond::getConstant() const
{
  return d_constant;
}
