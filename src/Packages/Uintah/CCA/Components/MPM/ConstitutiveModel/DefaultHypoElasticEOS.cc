
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
  double G = state->shearModulus;
  double p_n = state->pressure;

  // Calculate lambda
  double lambda = K - (2.0/3.0)*G;

  // Calculate pressure increment
  double delp = rateOfDeformation.Trace()*(lambda*delT);

  // Calculate pressure
  double p = p_n + delp;
  return p;
}
