// One of the derived Burn classes.  This particular
// class is used to specify the burn rate on the grid as a function of pressure.  

#include <Packages/Uintah/CCA/Components/HETransformation/PressureBurn.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

PressureBurn::PressureBurn(ProblemSpecP& ps)
{
  // Constructor

  ps->require("ThresholdTemp",thresholdTemp);
  ps->require("ThresholdPressure",thresholdPressure);
  ps->require("Enthalpy",Enthalpy);
  ps->require("BurnCoeff",BurnCoeff);
  ps->require("PressureExponent",pressureExponent);
  ps->require("ReferencePressure",refPressure);

  d_burnable = true;  

}

PressureBurn::~PressureBurn()
{
  // Destructor

}

void PressureBurn::computeBurn(double gasTemperature,
			     double gasPressure,
			     double materialMass,
			     double /*materialTemperature */,
			     double &burnedMass,
			     double &releasedHeat,
			     double &delT,
			     double &surfaceArea)
{
  if ((gasTemperature > thresholdTemp) && (gasPressure > thresholdPressure)) {
    burnedMass = surfaceArea * BurnCoeff * pow(gasPressure/refPressure,pressureExponent);
  }
  else {
    burnedMass = 0;
  }

  burnedMass *= delT;
  if(burnedMass > materialMass){
    burnedMass = materialMass;
  }
  releasedHeat = burnedMass * Enthalpy;
}

double PressureBurn::getThresholdTemperature()
{
   return thresholdTemp;
}
