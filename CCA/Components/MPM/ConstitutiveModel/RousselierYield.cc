#include "RousselierYield.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>
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

double
RousselierYield::evalDerivativeWRTPlasticityScalar(double trSig,
						   double porosity,
						   double sigY,
						   double dsigYdV)
{
  ostringstream desc;
  desc << "Rousselier Yield::evalDerivativeWRTPlasticityScalar not yet "
       << "implemented.  Inputs: " << trSig << ", " << porosity << ", " 
       << sigY << ", " << dsigYdV << endl;
  throw ProblemSetupException(desc.str());
  //return 0.0;
}

double
RousselierYield::evalDerivativeWRTPorosity(double trSig,
					   double porosity,
					   double sigY)
{
  ostringstream desc;
  desc << "Rousselier Yield::evalDerivativeWRTPorosity not yet implemented"
       << "Inputs: " << trSig << ", " << porosity << ", " << sigY << endl;
  throw ProblemSetupException(desc.str());
  //return 0.0;
}

inline double
RousselierYield::computePorosityFactor_h1(double sigma_f_sigma,
					  double tr_f_sigma,
					  double porosity,
					  double sigma_Y,
					  double A)
{
  return (1.0-porosity)*tr_f_sigma + A*sigma_f_sigma/((1.0-porosity)*sigma_Y);
}

inline double
RousselierYield::computePlasticStrainFactor_h2(double sigma_f_sigma,
					       double porosity,
					       double sigma_Y)
{
  return sigma_f_sigma/((1.0-porosity)*sigma_Y);
}


void 
RousselierYield::computeTangentModulus(const TangentModulusTensor& Ce,
				       const Matrix3& f_sigma, 
				       double f_q1, 
				       double f_q2,
				       double h_q1,
				       double h_q2,
				       TangentModulusTensor& Cep)
{
  double fqhq = f_q1*h_q1 + f_q2*h_q2;
  Matrix3 Cr(0.0), rC(0.0);
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      Cr(ii+1,jj+1) = 0.0;
      rC(ii+1,jj+1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          Cr(ii+1,jj+1) += Ce(ii,jj,kk,ll)*f_sigma(kk+1,ll+1);
          rC(ii+1,jj+1) += f_sigma(kk+1,ll+1)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii+1,jj+1)*f_sigma(ii+1,jj+1);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
	    Cr(ii+1,jj+1)*rC(kk+1,ll+1)/(-fqhq + rCr);
	}  
      }  
    }  
  }  
}

void 
RousselierYield::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
					       const Matrix3& sigma, 
					       double sigY,
					       double dsigYdep,
					       double porosity,
					       double voidNuclFac,
					       TangentModulusTensor& Cep)
{
  // Calculate the derivative of the yield function wrt sigma
  Matrix3 f_sigma;
  evalDerivOfYieldFunction(sigma, sigY, porosity, f_sigma);

  // Calculate derivative wrt porosity 
  double trSig = sigma.Trace();
  double f_q1 = evalDerivativeWRTPorosity(trSig, porosity, sigY);

  // Calculate derivative wrt plastic strain 
  double f_q2 = evalDerivativeWRTPlasticityScalar(trSig, porosity, sigY,
                                                  dsigYdep);
  // Compute h_q1
  double sigma_f_sigma = sigma.Contract(f_sigma);
  double tr_f_sigma = f_sigma.Trace();
  double h_q1 = computePorosityFactor_h1(sigma_f_sigma, tr_f_sigma, porosity,
                                         sigY, voidNuclFac);

  // Compute h_q2
  double h_q2 = computePlasticStrainFactor_h2(sigma_f_sigma, porosity, sigY);

  // Calculate elastic-plastic tangent modulus
  computeTangentModulus(Ce, f_sigma, f_q1, f_q2, h_q1, h_q2, Cep);
}

