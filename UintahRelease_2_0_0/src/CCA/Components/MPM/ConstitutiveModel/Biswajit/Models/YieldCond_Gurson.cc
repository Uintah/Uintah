/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "YieldCond_Gurson.h"        
#include <Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
#include <cmath>


using namespace std;
using namespace Uintah;
using namespace UintahBB;

YieldCond_Gurson::YieldCond_Gurson(Uintah::ProblemSpecP& ps)
{
  ps->require("q1",d_CM.q1);
  ps->require("q2",d_CM.q2);
  ps->require("q3",d_CM.q3);
  ps->require("k",d_CM.k);
  ps->require("f_c",d_CM.f_c);
}
         
YieldCond_Gurson::YieldCond_Gurson(const YieldCond_Gurson* cm)
{
  d_CM.q1 = cm->d_CM.q1;
  d_CM.q2 = cm->d_CM.q2;
  d_CM.q3 = cm->d_CM.q3;
  d_CM.k = cm->d_CM.k;
  d_CM.f_c = cm->d_CM.f_c;
}
         
YieldCond_Gurson::~YieldCond_Gurson()
{
}

void YieldCond_Gurson::outputProblemSpec(Uintah::ProblemSpecP& ps)
{
  ProblemSpecP yield_ps = ps->appendChild("plastic_yield_condition");
  yield_ps->setAttribute("type","gurson");
  yield_ps->appendElement("q1",d_CM.q1);
  yield_ps->appendElement("q2",d_CM.q2);
  yield_ps->appendElement("q3",d_CM.q3);
  yield_ps->appendElement("k",d_CM.k);
  yield_ps->appendElement("f_c",d_CM.f_c);

}
         
double 
YieldCond_Gurson::evalYieldCondition(const double sigEqv,
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

  double maxincosh = 30.0;
  double incosh = 0.5*q2*traceSig/sigFlow;
  if (fabs(incosh) > 30.0) incosh = copysign(maxincosh,incosh);
  ASSERT(sigFlow != 0);
  double a = 1.0 + q3*fStar*fStar;
  double b = 2.0*q1*fStar*cosh(incosh);
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
YieldCond_Gurson::evalDerivOfYieldFunction(const Matrix3& sig,
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

  double maxinsinh = 30.0;
  double insinh = 0.5*d_CM.q2*trSig/sigY;
  if (fabs(insinh) > 30.0) insinh = copysign(maxinsinh,insinh);
  derivative = sigDev*3.0 + 
               I*((d_CM.q1*d_CM.q2*fStar*sigY)*sinh(insinh));
}

void 
YieldCond_Gurson::evalDevDerivOfYieldFunction(const Matrix3& sig,
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
YieldCond_Gurson::evalDerivativeWRTPlasticityScalar(double trSig,
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
  double maxinsinh = 30.0;
  if (fabs(BsigY) > 30.0) BsigY = copysign(maxinsinh,BsigY);
  double dPhidV = -ABdsigYdV*sinh(BsigY) + 2.0*sigYdsigYdV*(A*cosh(BsigY)-C);
  return dPhidV;
}

double
YieldCond_Gurson::evalDerivativeWRTPorosity(double trSig,
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
  double maxincosh = 30.0;
  double incosh = 0.5*d_CM.q2*trSig/sigY;
  if (fabs(incosh) > 30.0) incosh = copysign(maxincosh,incosh);
  double a = 2.0*d_CM.q3*fStar;
  double b = 2.0*d_CM.q1*cosh(incosh);
  double dPhi_dfStar = (b - a)*sigY*sigY;
  
  return (dPhi_dfStar*dfStar_df);
}

inline double
YieldCond_Gurson::computePorosityFactor_h1(double sigma_f_sigma,
                                      double tr_f_sigma,
                                      double porosity,
                                      double sigma_Y,
                                      double A)
{
  return (1.0-porosity)*tr_f_sigma + A*sigma_f_sigma/((1.0-porosity)*sigma_Y);
}

inline double
YieldCond_Gurson::computePlasticStrainFactor_h2(double sigma_f_sigma,
                                           double porosity,
                                           double sigma_Y)
{
  return sigma_f_sigma/((1.0-porosity)*sigma_Y);
}

double 
YieldCond_Gurson::evalYieldCondition(const Matrix3& xi,
                                const ModelState* state)
{
  // Get the state data
  double porosity = state->porosity;
  double sigy = state->yieldStress;
  double trSig = 3.0*state->pressure;
  double xiNormSq = xi.NormSquared();

  double q1 = d_CM.q1;
  double q2 = d_CM.q2;
  double q3 = d_CM.q3;
  double k = d_CM.k;
  double phi_c = d_CM.f_c;
  
  double phiStar = porosity;
  if (porosity > phi_c) phiStar = phi_c + k*(porosity - phi_c);

  double maxincosh = 30.0;
  ASSERT(sigy != 0);
  double incosh = 0.5*q2*trSig/sigy;
  if (fabs(incosh) > 30.0) incosh = copysign(maxincosh,incosh);
  double a = 2.0*q1*phiStar*cosh(incosh);
  double b = 1.0 + q3*phiStar*phiStar;
  double sigYSq = sigy*sigy;
  double Phi = 1.5*xiNormSq/sigYSq + a - b;
  return Phi;
}

/*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
void 
YieldCond_Gurson::eval_df_dsigma(const Matrix3& xi,
                            const ModelState* state,
                            Matrix3& df_dsigma)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double trSig = 3.0*state->pressure;
  double porosity = state->porosity;

  Matrix3 One; One.Identity();
  double q1 = d_CM.q1;
  double q2 = d_CM.q2;
  double k = d_CM.k;
  double phi_c = d_CM.f_c;
  
  double phiStar = porosity;
  if (porosity > phi_c) phiStar = phi_c + k*(porosity - phi_c);

  double maxinsinh = 30.0;
  double insinh = 0.5*q2*trSig/sigy;
  if (fabs(insinh) > 30.0) insinh = copysign(maxinsinh,insinh);
  double a = 3.0/(sigy*sigy);
  double b = q1*q2*phiStar*sinh(insinh);
  df_dsigma = xi*a + One*b;
  return;
}

/*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
    where \f$s\f$ is deviatoric part of Cauchy stress and 
    \f$\beta\f$ is the backstress */
void 
YieldCond_Gurson::eval_df_dxi(const Matrix3& xi,
                         const ModelState* state,
                         Matrix3& df_dxi)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double a = 3.0/(sigy*sigy);
  df_dxi = xi*a;
  return;
}

/* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
void 
YieldCond_Gurson::eval_df_ds_df_dbeta(const Matrix3& xi,
                                 const ModelState* state,
                                 Matrix3& df_ds,
                                 Matrix3& df_dbeta)
{
  eval_df_dxi(xi, state, df_ds);
  df_dbeta = df_ds*(-1.0); 
  return;
}

/*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
double 
YieldCond_Gurson::eval_df_dep(const Matrix3& xi,
                         const double& dsigy_dep,
                         const ModelState* state)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double trSig = 3.0*state->pressure;
  double porosity = state->porosity;
  double xiNormSq = xi.NormSquared();

  double q1 = d_CM.q1;
  double q2 = d_CM.q2;
  double k = d_CM.k;
  double phi_c = d_CM.f_c;
  
  double phiStar = porosity;
  if (porosity > phi_c) phiStar = phi_c + k*(porosity - phi_c);

  double maxinsinh = 30.0;
  double insinh = 0.5*q2*trSig/sigy;
  if (fabs(insinh) > 30.0) insinh = copysign(maxinsinh,insinh);
  double a = (3.0*xiNormSq)/(sigy*sigy*sigy);
  double b = q1*q2*phiStar*sinh(insinh)/(sigy*sigy);

  return -(a+b)*dsigy_dep;
}

/*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
double 
YieldCond_Gurson::eval_df_dphi(const Matrix3& xi,
                          const ModelState* state)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double trSig = 3.0*state->pressure;
  double porosity = state->porosity;

  double q1 = d_CM.q1;
  double q2 = d_CM.q2;
  double q3 = d_CM.q3;
  double k = d_CM.k;
  double phi_c = d_CM.f_c;
  
  double phiStar = porosity;
  double dphiStar_dphi = 1.0;
  if (porosity > phi_c) {
    phiStar = phi_c + k*(porosity - phi_c);
    dphiStar_dphi = k;
  }

  double maxincosh = 30.0;
  double incosh = 0.5*q2*trSig/sigy;
  if (fabs(incosh) > maxincosh) incosh = copysign(maxincosh,incosh);
  double a = 2.0*q1*dphiStar_dphi*cosh(incosh);
  double b = 2.0*q3*phiStar*dphiStar_dphi;

  return (a - b);
}

/*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
double 
YieldCond_Gurson::eval_h_alpha(const Matrix3& xi,
                          const ModelState* state)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double phi = state->porosity;
  ASSERT(phi != 1);
  double p = state->pressure;
  double trBetaHat = state->backStress.Trace();

  Matrix3 One; One.Identity();
  Matrix3 xi_hat = xi + One*(p - 1.0/3.0*trBetaHat);

  Matrix3 df_dsigma(0.0); 
  eval_df_dsigma(xi, state, df_dsigma);

  double numer = xi_hat.Contract(df_dsigma);
  double denom = (1.0 - phi)*sigy;
  return numer/denom;
}

/*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
double 
YieldCond_Gurson::eval_h_phi(const Matrix3& xi,
                        const double& factorA,
                        const ModelState* state)
{
  double sigy = state->yieldStress;
  ASSERT(sigy != 0);
  double phi = state->porosity;
  ASSERT(phi != 1);
  double p = state->pressure;

  Matrix3 df_dsigma(0.0); 
  eval_df_dsigma(xi, state, df_dsigma);

  double tr_df_dsigma = df_dsigma.Trace();
  double a = (1.0 - phi)*tr_df_dsigma;

  Matrix3 One; One.Identity();
  double trBetaHat = state->backStress.Trace();
  Matrix3 xi_hat = xi + One*(p - 1.0/3.0*trBetaHat);

  double numer = xi_hat.Contract(df_dsigma);
  double denom = (1.0 - phi)*sigy;
  double b = factorA*numer/denom;

  return (a + b);
}

void 
YieldCond_Gurson::computeTangentModulus(const TangentModulusTensor& Ce,
                                   const Matrix3& f_sigma, 
                                   double f_q1, 
                                   double f_q2,
                                   double h_q1,
                                   double h_q2,
                                   TangentModulusTensor& Cep)
{
  //cerr << getpid() << " Ce = " << Ce;
  //cerr << getpid() << " f_sigma = " << f_sigma << endl;
  //cerr << getpid() << " f_q1 = " << f_q1 << " f_q2 = " << f_q2 << endl;
  //cerr << getpid() << " h_q1 = " << h_q1 << " h_q2 = " << h_q2 << endl << endl;
  double fqhq = f_q1*h_q1 + f_q2*h_q2;
  Matrix3 Cr(0.0), rC(0.0);
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      Cr(ii,jj) = 0.0;
      rC(ii,jj) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          double Ce1 = Ce(ii,jj,kk,ll);
          double Ce2 = Ce(kk,ll,ii,jj);
          double fs = f_sigma(kk,ll);
          Cr(ii,jj) += Ce1*fs;
          rC(ii,jj) += fs*Ce2;
        }
      }
      rCr += rC(ii,jj)*f_sigma(ii,jj);
    }
  }
  double rCr_fqhq = rCr - fqhq;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
            Cr(ii,jj)*rC(kk,ll)/rCr_fqhq;
        }  
      }  
    }  
  }  
}

void 
YieldCond_Gurson::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
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

