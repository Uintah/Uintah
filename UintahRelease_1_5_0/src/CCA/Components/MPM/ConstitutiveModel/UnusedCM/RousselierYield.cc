/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "RousselierYield.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <sstream>
#include <cmath>

using namespace Uintah;

RousselierYield::RousselierYield(ProblemSpecP& ps)
{
  ps->require("D",d_constant.D);
  ps->require("sig_1",d_constant.sig_1);
}
         
RousselierYield::RousselierYield(const RousselierYield* cm)
{
  d_constant.D = cm->d_constant.D;
  d_constant.sig_1 = cm->d_constant.sig_1;
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
      Cr(ii,jj) = 0.0;
      rC(ii,jj) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cr(ii,jj) += Ce(ii,jj,kk,ll)*f_sigma(kk,ll);
          rC(ii,jj) += f_sigma(kk,ll)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii,jj)*f_sigma(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
            Cr(ii,jj)*rC(kk,ll)/(-fqhq + rCr);
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

