
#include "JohnsonCookPlastic.h"	
#include <math.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

JohnsonCookPlastic::JohnsonCookPlastic(ProblemSpecP& ps)
{
  ps->require("A",d_initialData.A);
  ps->require("B",d_initialData.B);
  ps->require("C",d_initialData.C);
  ps->require("n",d_initialData.n);
  ps->require("m",d_initialData.m);
}
	 
JohnsonCookPlastic::~JohnsonCookPlastic()
{
}
	 
double 
JohnsonCookPlastic::computeFlowStress(const Matrix3& rateOfDeformation,
                                      const Matrix3& ,
                                      const double& temperature,
                                      const double& delT,
                                      const MPMMaterial* matl,
                                      const double& tolerance,
                                      double& plasticStrain)
{
  double plasticStrainRate = sqrt(rateOfDeformation.NormSquared()*2.0/3.0);
  plasticStrain += plasticStrainRate*delT;

  return evaluateFlowStress(plasticStrain, plasticStrainRate, temperature, matl, tolerance);
}

double 
JohnsonCookPlastic::evaluateFlowStress(const double& ep, 
				       const double& epdot,
				       const double& T,
                                       const MPMMaterial* matl,
                                       const double& tolerance)
{
  double strainPart = d_initialData.A + d_initialData.B*pow(ep,d_initialData.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.C);
  else
    strainRatePart = 1.0 + d_initialData.C*log(epdot);
  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  double m = d_initialData.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  if (fabs(Tstar) < tolerance) Tstar = 0.0;
  if (Tstar < 0.0) {
    cerr << " ep = " << ep << " Strain Part = " << strainPart << endl;
    cerr << "epdot = " << epdot << " Strain Rate Part = " << strainRatePart << endl;
    cerr << "Tstar = " << Tstar << " T = " << T << " Tr = " << Tr << " Tm = " << Tm << endl;
  }
  ASSERT(Tstar > -tolerance);
  double tm = pow(Tstar,m);
  double tempPart = 1.0 - tm;
  return (strainPart*strainRatePart*tempPart);
}

