// One of the derived Burn classes.  This particular
// class is used when no burn is desired.  

#include "SimpleBurn.h"

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

  d_burnable = true;  

}

SimpleBurn::~SimpleBurn()
{
  // Destructor

}

void SimpleBurn::computeBurn(double gasTemperature,
			       double gasPressure,
			       double materialMass,
			       double materialTemperature,
			       double &burnedMass,
			       double &releasedHeat)
{
  if ((gasTemperature > thresholdTemp) || (gasPressure > thresholdPressure)) 
    {
      burnedMass = BurnCoeff * pow(gasPressure,0.778);
      releasedHeat = burnedMass * Enthalpy;
    }
  else
    {
      burnedMass = 0;
      releasedHeat = 0;
    }

  if (burnedMass > materialMass)
    {
      burnedMass = materialMass;
      releasedHeat = burnedMass * Enthalpy;
    }
}

















