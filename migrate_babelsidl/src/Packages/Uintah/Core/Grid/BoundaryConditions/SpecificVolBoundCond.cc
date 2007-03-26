#include <Packages/Uintah/Core/Grid/BoundaryConditions/SpecificVolBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SpecificVolBoundCond::SpecificVolBoundCond(ProblemSpecP& ps, std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "SpecificVol";
  ps->require("value",d_value);
}

SpecificVolBoundCond::~SpecificVolBoundCond()
{
}

SpecificVolBoundCond* SpecificVolBoundCond::clone()
{
  return scinew SpecificVolBoundCond(*this);
}
