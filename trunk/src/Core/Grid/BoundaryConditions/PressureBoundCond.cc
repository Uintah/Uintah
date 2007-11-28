#include <Core/Grid/BoundaryConditions/PressureBoundCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
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
