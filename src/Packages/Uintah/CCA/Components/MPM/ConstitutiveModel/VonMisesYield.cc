#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/VonMisesYield.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
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

