
#include "VonMisesYield.h"	
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

VonMisesYield::VonMisesYield(ProblemSpecP&)
{
}
	 
VonMisesYield::~VonMisesYield()
{
}
	 
double 
VonMisesYield::evalYieldCondition(const double sigEqv,
                                  const double sigFlow,
                                  const double,
                                  const double)
{
  return (sigEqv-sigFlow);
}

