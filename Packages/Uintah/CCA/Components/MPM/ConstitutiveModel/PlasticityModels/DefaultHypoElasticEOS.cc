
#include "DefaultHypoElasticEOS.h"
#include <cmath>

using namespace Uintah;
using namespace SCIRun;

DefaultHypoElasticEOS::DefaultHypoElasticEOS()
{
} 

DefaultHypoElasticEOS::DefaultHypoElasticEOS(ProblemSpecP&)
{
} 
	 
DefaultHypoElasticEOS::DefaultHypoElasticEOS(const DefaultHypoElasticEOS*)
{
} 
	 
DefaultHypoElasticEOS::~DefaultHypoElasticEOS()
{
}


void DefaultHypoElasticEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","default_hypo");
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
  double kappa = state->bulkModulus;
  double p_n = state->pressure;

  // Calculate pressure increment
  double delp = rateOfDeformation.Trace()*(kappa*delT);

  // Calculate pressure
  double p = p_n + delp;
  return p;
}

double 
DefaultHypoElasticEOS::eval_dp_dJ(const MPMMaterial* matl,
                                  const double& detF, 
                                  const PlasticityState* state)
{
  return (state->bulkModulus/detF);
}
