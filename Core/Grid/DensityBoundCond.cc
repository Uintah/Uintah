#include <Packages/Uintah/Core/Grid/DensityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

DensityBoundCond::DensityBoundCond(ProblemSpecP& ps, std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "Density";
  ps->require("value",d_value);
}

DensityBoundCond::~DensityBoundCond()
{
}

DensityBoundCond* DensityBoundCond::clone()
{
  return scinew DensityBoundCond(*this);
}

