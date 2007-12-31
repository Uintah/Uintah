
#include "HyperElasticEOS.h"
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

HyperElasticEOS::HyperElasticEOS()
{
} 

HyperElasticEOS::HyperElasticEOS(ProblemSpecP&)
{
} 
	 
HyperElasticEOS::HyperElasticEOS(const HyperElasticEOS*)
{
} 
	 
HyperElasticEOS::~HyperElasticEOS()
{
}


void HyperElasticEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","default_hypo");
}
	 

//////////
// Calculate the pressure using the elastic constitutive equation
double 
HyperElasticEOS::computePressure(const MPMMaterial* matl,
				       const PlasticityState* state,
				       const Matrix3& ,
				       const Matrix3& rateOfDeformation,
				       const double& delT)
{
  double rho_0 = matl->getInitialDensity();
  double rho = state->density;
  double J = rho_0/rho;
  double kappa = state->bulkModulus;

  double p = 0.5*kappa*(J - 1.0/J);
  return p;
}

double 
HyperElasticEOS::eval_dp_dJ(const MPMMaterial* matl,
                                  const double& detF, 
                                  const PlasticityState* state)
{
  double J = detF;
  double kappa = state->bulkModulus;

  double dpdJ = 0.5*kappa*(1.0 + 1.0/(J*J));
  return dpdJ;
}
