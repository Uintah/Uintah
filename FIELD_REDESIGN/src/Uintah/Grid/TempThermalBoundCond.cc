#include <Uintah/Grid/TempThermalBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

TempThermalBoundCond::TempThermalBoundCond(double& t) : d_temp(t)
{
}

TempThermalBoundCond::TempThermalBoundCond(ProblemSpecP& ps)
{
  ps->require("temperature",d_temp);
}

TempThermalBoundCond::~TempThermalBoundCond()
{
}

double TempThermalBoundCond::getTemp() const
{
  return d_temp;
}

std::string TempThermalBoundCond::getType() const
{
  return "Temperature";
}
