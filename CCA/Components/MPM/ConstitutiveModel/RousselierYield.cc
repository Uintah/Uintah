#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/RousselierYield.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

using namespace Uintah;

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
				    const double porosity,
                                    double& sig)
{
  double D = d_constant.D;
  double sig1 = d_constant.sig_1;
  double f = porosity;

  sig = (1.0-f)*sigFlow - 
         D*sig1*f*(1.0-f)*exp((1.0/3.0)*traceSig/((1.0-f)*sig1));
  double Phi = sigEqv - sig; 
  sig = sqrt(1.5*sig);

  return Phi;
}

void 
RousselierYield::evalDerivOfYieldFunction(const Matrix3& sig,
					  const double sigFlow,
					  const double f,
					  Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  double sigEqv = sqrt((sigDev.NormSquared())*1.5);

  double D = d_constant.D;
  double sig1 = d_constant.sig_1;
  double con = trSig/(3.0*(1.0-f)*sig1);

  derivative = sigDev*(1.5/sigEqv) + I*(D*f*exp(con)/3.0);
}

/*! \warning Derivative is taken assuming sig_eq^2 - sig_Y^2 = 0 form.
             This is needed for the HypoElasticPlastic algorithm.  Needs
             to be more generalized if possible. */
void 
RousselierYield::evalDevDerivOfYieldFunction(const Matrix3& sig,
					     const double ,
					     const double ,
					     Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  //double sigEqv = sqrt((sigDev.NormSquared())*1.5);
  //derivative = sigDev*(1.5/sigeqv);
  derivative = sigDev*3.0;
}
