// One of the derived Burn classes.  This particular
// class is used when no burn is desired.  

#include <Packages/Uintah/CCA/Components/HETransformation/IgnitionCombustion.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
using namespace std;

using namespace Uintah;

IgnitionCombustion::IgnitionCombustion(ProblemSpecP& ps)
{
  // Constructor

  ps->require("ThresholdTemp",thresholdTemp);
  ps->require("ThresholdPressure",thresholdPressure);
  ps->require("Enthalpy",Enthalpy);
  ps->require("BurnCoeff",BurnCoeff);
  ps->require("activationEnergy",activationEnergy);
  ps->require("preExponent",preExponent);

  d_burnable = true;  

}

IgnitionCombustion::~IgnitionCombustion()
{
  // Destructor
}

void IgnitionCombustion::computeBurn(double gasTemperature,
				     double gasPressure,
				     double materialMass,
				     double materialTemperature,
				     double &burnedMass,
				     double &releasedHeat,
				     double &delT,
				     double &surfaceArea)
{
  
  if ((gasTemperature > thresholdTemp) && (gasPressure > thresholdPressure)) {
      burnedMass = surfaceArea * BurnCoeff * pow(gasPressure,0.778);
      cout << "ignited  " << flush;
  }
  else {
    burnedMass = preExponent * exp(-activationEnergy/materialTemperature) * 
      surfaceArea;
  }

  burnedMass *= delT;
  if(burnedMass > materialMass){
    burnedMass = materialMass;
  }
  releasedHeat = burnedMass * Enthalpy;
}

double IgnitionCombustion::getThresholdTemperature()
{
   return thresholdTemp;
}
