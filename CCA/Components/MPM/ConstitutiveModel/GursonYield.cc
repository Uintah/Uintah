#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/GursonYield.h>	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <math.h>

using namespace std;
using namespace Uintah;

GursonYield::GursonYield(ProblemSpecP& ps)
{
  ps->require("q1",d_constant.q1);
  ps->require("q2",d_constant.q2);
  ps->require("q3",d_constant.q3);
  ps->require("k",d_constant.k);
  ps->require("f_c",d_constant.f_c);
}
	 
GursonYield::~GursonYield()
{
}
	 
double 
GursonYield::evalYieldCondition(const double sigEqv,
				const double sigFlow,
				const double traceSig,
				const double porosity,
                                double& sig)
{
  double q1 = d_constant.q1;
  double q2 = d_constant.q2;
  double q3 = d_constant.q3;
  double k = d_constant.k;
  double f_c = d_constant.f_c;
  
  double fStar = porosity;
  if (porosity > f_c) fStar = f_c + k*(porosity - f_c);

  ASSERT(sigFlow != 0);
  double a = 1.0 + q3*fStar*fStar;
  double b = 2.0*q1*fStar*cosh(0.5*q2*traceSig/sigFlow);
  double aminusb = a - b;
  double Phi = -1.0;
  double sigYSq = sigFlow*sigFlow;
  if (aminusb < 0.0) {
    Phi = sigEqv*sigEqv - sigYSq;
    sig = sigFlow;
  } else {
    sig = sigYSq*(a-b);
    Phi = sigEqv*sigEqv - sig;
    sig = sqrt(sig);
  }
  return Phi;
}

void 
GursonYield::evalDerivOfYieldFunction(const Matrix3& sig,
				      const double sigFlow,
				      const double porosity,
				      Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  //double sigEqv = sqrt((sigDev.NormSquared())*1.5);

  double q1 = d_constant.q1;
  double q2 = d_constant.q2;
  double k = d_constant.k;
  double f_c = d_constant.f_c;
  double fStar = porosity;
  if (porosity > f_c) fStar = f_c + k*(porosity - f_c);

  derivative = sigDev*3.0 + 
               I*((q1*q2*fStar*sigFlow)*sinh(0.5*q2*trSig/sigFlow));
}

void 
GursonYield::evalDevDerivOfYieldFunction(const Matrix3& sig,
					 const double ,
					 const double ,
					 Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  derivative = sigDev*3.0;
}
