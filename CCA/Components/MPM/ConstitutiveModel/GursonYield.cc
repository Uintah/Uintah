#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/GursonYield.h>	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

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
				const double porosity)
{
  double q1 = d_constant.q1;
  double q2 = d_constant.q2;
  double q3 = d_constant.q3;
  double k = d_constant.k;
  double f_c = d_constant.f_c;
  
  double fStar = porosity;
  if (porosity > f_c) fStar = f_c + k*(porosity - f_c);

  double Phi = sigEqv*sigEqv/sigFlow + 
    2.0*q1*fStar*cosh(0.5*q2*traceSig/sigFlow) - (1.0+q3*fStar*fStar);

  return Phi;
}

