
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DefaultHypoElasticEOS.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

DefaultHypoElasticEOS::DefaultHypoElasticEOS(ProblemSpecP&)
{
} 
	 
DefaultHypoElasticEOS::~DefaultHypoElasticEOS()
{
}
	 

//////////
// Calculate the pressure using the elastic constitutive equation
double 
DefaultHypoElasticEOS::computePressure(const MPMMaterial* ,
				       const PlasticityState* state,
				       const Matrix3& ,
				       const Matrix3& rateOfDeformation,
				       const double& delT)
{

  // Get the state data
  double K = state->bulkModulus;

  // Calculate pressure
  double p = rateOfDeformation.Trace()*(K*delT);
  return p;
}
