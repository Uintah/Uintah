#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

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

