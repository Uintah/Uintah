#include <Core/Grid/BoundaryConditions/SpecificVolBoundCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <SCIRun/Core/Malloc/Allocator.h>

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
