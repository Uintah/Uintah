#include <Uintah/Grid/PressureBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

PressureBoundCond::PressureBoundCond(double& p)
{
  d_press = p;
}

PressureBoundCond::PressureBoundCond(ProblemSpecP& ps)
{
  ps->require("pressure",d_press);
}

PressureBoundCond::~PressureBoundCond()
{
}

double PressureBoundCond::getPressure() const
{
  return d_press;
}

std::string PressureBoundCond::getType() const
{
  return "Pressure";
}

