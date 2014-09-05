
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
Matrix3 
DefaultHyperElasticEOS::computePressure(const MPMMaterial* ,
					const double& bulk,
					const double& ,
					const Matrix3& deformGrad,
					const Matrix3& ,
					const Matrix3& ,
					const double& ,
					const double& ,
					const double& )
{

  // Calculate lambda
  double J = deformGrad.Determinant();

  // Calculate pressure
  double p = 0.5*bulk*(J - 1.0/J);

  // Calculate the hydrostatic part of stress tensor
  Matrix3 one; one.Identity();
  return (one*p);
}
