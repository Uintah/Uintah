#include <Uintah/Grid/FluxThermalBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>

FluxThermalBoundCond::FluxThermalBoundCond(double& f) : d_flux(f)
{
}

FluxThermalBoundCond::FluxThermalBoundCond(ProblemSpecP& ps)
{
  ps->require("heat_flux",d_flux);
  

}
FluxThermalBoundCond::~FluxThermalBoundCond()
{
}


double FluxThermalBoundCond::getFlux() const
{
  return d_flux;
}

std::string FluxThermalBoundCond::getType() const
{
  return "Flux";
}
