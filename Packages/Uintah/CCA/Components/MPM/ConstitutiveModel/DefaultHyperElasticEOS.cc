
#include "DefaultHyperElasticEOS.h"	
#include <math.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

DefaultHyperElasticEOS::DefaultHyperElasticEOS(ProblemSpecP& ps)
{
} 
	 
DefaultHyperElasticEOS::~DefaultHyperElasticEOS()
{
}
	 

//////////
// Calculate the pressure using the elastic constitutive equation
double 
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
  return p;
}
