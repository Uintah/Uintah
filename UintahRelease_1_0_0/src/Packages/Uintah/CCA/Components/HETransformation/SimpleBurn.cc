// One of the derived Burn classes.  This particular
// class is used when no burn is desired.  

#include <Packages/Uintah/CCA/Components/HETransformation/SimpleBurn.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

SimpleBurn::SimpleBurn(ProblemSpecP& ps)
{
  // Constructor

  ps->require("ThresholdTemp",thresholdTemp);
  ps->require("ThresholdPressure",thresholdPressure);
  ps->require("Enthalpy",Enthalpy);
  ps->require("BurnCoeff",BurnCoeff);
  ps->require("refPressure",refPressure);

  d_burnable = true;  

}

SimpleBurn::~SimpleBurn()
{
  // Destructor

}

void SimpleBurn::computeBurn(double gasTemperature,
			     double gasPressure,
			     double materialMass,
			     double /*materialTemperature */,
			     double &burnedMass,
			     double &releasedHeat,
			     double &delT,
			     double &surfaceArea)
{
  if ((gasTemperature > thresholdTemp) && (gasPressure > thresholdPressure)) {
    burnedMass = surfaceArea * BurnCoeff * pow((gasPressure/refPressure),0.778);
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

double SimpleBurn::getThresholdTemperature()
{
   return thresholdTemp;
}
