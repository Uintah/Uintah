
#include "RousselierYield.h"	
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

RousselierYield::RousselierYield(ProblemSpecP& ps)
{
  ps->require("D",d_constant.D);
  ps->require("sig_1",d_constant.sig_1);
}
	 
RousselierYield::~RousselierYield()
{
}
	 
double 
RousselierYield::evalYieldCondition(const double sigEqv,
                                  const double sigFlow,
                                  const double traceSig,
                                  const double porosity)
{
  double D = d_constant.D;
  double sig1 = d_constant.sig_1;
  double f = porosity;
  
  double Phi = sigEqv/(1.0-f) + D*sig1*f*exp((1.0/3.0)*traceSig/((1.0-f)*sig1))
               - sigFlow;

  return Phi;
}

