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
  ps->require("q1",d_CM.q1);
  ps->require("q2",d_CM.q2);
  ps->require("q3",d_CM.q3);
  ps->require("k",d_CM.k);
  ps->require("f_c",d_CM.f_c);
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
  double q1 = d_CM.q1;
  double q2 = d_CM.q2;
  double q3 = d_CM.q3;
  double k = d_CM.k;
  double f_c = d_CM.f_c;
  
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
				      const double sigY,
				      const double f,
				      Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  //double sigEqv = sqrt((sigDev.NormSquared())*1.5);

  double fStar = f;
  if (f > d_CM.f_c) fStar = d_CM.f_c + d_CM.k*(f - d_CM.f_c);

  derivative = sigDev*3.0 + 
               I*((d_CM.q1*d_CM.q2*fStar*sigY)*sinh(0.5*d_CM.q2*trSig/sigY));
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


double
GursonYield::evalDerivativeWRTPlasticityScalar(double trSig,
                                               double porosity,
                                               double sigY,
                                               double dsigYdV)
{
  // Calculate fStar
  double fStar = porosity;
  if (porosity > d_CM.f_c) fStar = d_CM.f_c + d_CM.k*(porosity - d_CM.f_c);

  // Calculate A, B, C
  double A = 2.0*d_CM.q1*fStar;
  double B = 0.5*d_CM.q2*trSig;
  double C = 1.0 + d_CM.q3*fStar*fStar;

  // Calculate terms  
  double ABdsigYdV = A*B*dsigYdV;
  double sigYdsigYdV = sigY*dsigYdV;
  double BsigY = B/sigY;

  // Calculate derivative
  double dPhidV = -ABdsigYdV*sinh(BsigY) + 2.0*sigYdsigYdV*(A*cosh(BsigY)-C);
  return dPhidV;
}

double
GursonYield::evalDerivativeWRTPorosity(double trSig,
				       double porosity,
				       double sigY)
{
  double fStar = porosity;
  double dfStar_df = 1.0;
  if (porosity > d_CM.f_c) {
    fStar = d_CM.f_c + d_CM.k*(porosity - d_CM.f_c);
    dfStar_df = d_CM.k;
  }

  ASSERT(sigY != 0);
  double a = 2.0*d_CM.q3*fStar;
  double b = 2.0*d_CM.q1*cosh(0.5*d_CM.q2*trSig/sigY);
  double dPhi_dfStar = (b - a)*sigY*sigY;
  
  return (dPhi_dfStar*dfStar_df);
}

inline double
GursonYield::computePorosityFactor_h1(double sigma_f_sigma,
                                      double tr_f_sigma,
                                      double porosity,
                                      double sigma_Y,
                                      double A)
{
  return (1.0-porosity)*tr_f_sigma + A*sigma_f_sigma/((1.0-porosity)*sigma_Y);
}

inline double
GursonYield::computePlasticStrainFactor_h2(double sigma_f_sigma,
                                           double porosity,
                                           double sigma_Y)
{
  return sigma_f_sigma/((1.0-porosity)*sigma_Y);
}


void 
GursonYield::computeTangentModulus(const TangentModulusTensor& Ce,
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
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      Cr(ii1,jj1) = 0.0;
      rC(ii1,jj1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          double Ce1 = Ce(ii,jj,kk,ll);
          double Ce2 = Ce(kk,ll,ii,jj);
          double fs = f_sigma(kk1,ll+1);
          Cr(ii1,jj1) += Ce1*fs;
          rC(ii1,jj1) += fs*Ce2;
        }
      }
      rCr += rC(ii1,jj1)*f_sigma(ii1,jj1);
    }
  }
  double rCr_fqhq = rCr - fqhq;
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
	    Cr(ii1,jj1)*rC(kk1,ll+1)/rCr_fqhq;
	}  
      }  
    }  
  }  
}

void 
GursonYield::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
					   const Matrix3& sigma, 
					   double sigY,
					   double dsigYdep,
					   double porosity,
					   double voidNuclFac,
					   TangentModulusTensor& Cep)
{
  // Calculate the derivative of the yield function wrt sigma
  Matrix3 f_sigma(0.0);
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

