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
                                  const double,
                                  double& sig)
{
  sig = sigFlow;
  return (sigEqv-sigFlow);
}

void 
VonMisesYield::evalDerivOfYieldFunction(const Matrix3& sig,
					const double ,
					const double ,
					Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  double sigEqv = sqrt(1.5*sigDev.NormSquared());
  derivative = sigDev*(1.5/sigEqv);
}

