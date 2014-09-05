// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include <Packages/Uintah/CCA/Components/HETransformation/NullBurn.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

NullBurn::NullBurn(ProblemSpecP& /*ps*/)
{
  // Constructor
 
  d_burnable = false;

}

NullBurn::~NullBurn()
{
  // Destructor

}

void NullBurn::computeBurn(double /*gasTemperature */,
			   double /*gasPressure */,
			   double /*materialMass */,
			   double /*materialTemperature */,
			   double &burnedMass,
			   double &releasedHeat,
			   double &/*delT*/,
			   double &/*surfaceArea*/)
{
  burnedMass = 0;
  releasedHeat = 0;
}

double NullBurn::getThresholdTemperature()
{
   return 1.e15;
}

