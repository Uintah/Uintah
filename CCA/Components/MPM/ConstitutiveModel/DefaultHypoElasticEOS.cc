
#include "DefaultHypoElasticEOS.h"	
#include <math.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

DefaultHypoElasticEOS::DefaultHypoElasticEOS(ProblemSpecP& ps)
{
} 
	 
DefaultHypoElasticEOS::~DefaultHypoElasticEOS()
{
}
	 

//////////
// Calculate the pressure using the elastic constitutive equation
double 
DefaultHypoElasticEOS::computePressure(const MPMMaterial* ,
				       const double& bulk,
				       const double& shear,
				       const Matrix3& ,
				       const Matrix3& rateOfDeformation,
				       const Matrix3& ,
				       const double& ,
				       const double& ,
				       const double& delT)
{

  // Calculate lambda
  double lambda = bulk - (2.0/3.0)*shear;

  // Calculate pressure
  double p = -(rateOfDeformation.Trace()*lambda*delT);
  return p;
}
