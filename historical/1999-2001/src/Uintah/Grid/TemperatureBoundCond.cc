#include <Uintah/Grid/TemperatureBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

TemperatureBoundCond::TemperatureBoundCond(ProblemSpecP& ps,std::string& kind):
  BoundCond<double>(kind)
{
  d_type = "Temperature";
  ps->require("value",d_temp);
}

TemperatureBoundCond::~TemperatureBoundCond()
{
}

double TemperatureBoundCond::getValue() const
{
  return d_temp;
}

