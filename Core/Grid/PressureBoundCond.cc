#include <Packages/Uintah/Core/Grid/PressureBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

PressureBoundCond::PressureBoundCond(ProblemSpecP& ps,std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "Pressure";
  ps->require("value",d_value);
}

PressureBoundCond::~PressureBoundCond()
{
}

PressureBoundCond* PressureBoundCond::clone()
{
  return scinew PressureBoundCond(*this);
}
