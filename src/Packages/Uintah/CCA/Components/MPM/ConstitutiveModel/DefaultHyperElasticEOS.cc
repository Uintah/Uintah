
#include "DefaultHyperElasticEOS.h"	
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

DefaultHyperElasticEOS::DefaultHyperElasticEOS(ProblemSpecP&)
{
} 
	 
DefaultHyperElasticEOS::~DefaultHyperElasticEOS()
{
}
	 

//////////
// Calculate the pressure using the elastic constitutive equation
double 
DefaultHyperElasticEOS::computePressure(const MPMMaterial* ,
					const PlasticityState* state,
					const Matrix3& deformGrad,
					const Matrix3& ,
					const double& )
{
  // Get the state data
  double K = state->bulkModulus;

  // Calculate lambda
  double J = deformGrad.Determinant();

  // Calculate pressure
  double p = 0.5*K*(J - 1.0/J);
  return p;
}
