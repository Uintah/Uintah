#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

TemperatureBoundCond::TemperatureBoundCond(ProblemSpecP& ps,std::string& kind):
  BoundCond<double>(kind)
{
  d_type = "Temperature";
  ps->require("value",d_value);
}

TemperatureBoundCond::~TemperatureBoundCond()
{
}

TemperatureBoundCond* TemperatureBoundCond::clone()
{
  return new TemperatureBoundCond(*this);
}
