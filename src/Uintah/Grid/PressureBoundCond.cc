#include <Uintah/Grid/PressureBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

PressureBoundCond::PressureBoundCond(ProblemSpecP& ps,std::string& kind) 
  : BoundCond<double>(kind)
{
  d_type = "Pressure";
  ps->require("value",d_press);
}

PressureBoundCond::~PressureBoundCond()
{
}

double PressureBoundCond::getValue() const
{
  return d_press;
}

