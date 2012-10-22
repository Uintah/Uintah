/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include "YieldCondition.h"


using namespace Uintah;

YieldCondition::YieldCondition()
{
}

YieldCondition::~YieldCondition()
{
}
         

/*! Compute continuum elastic-plastic tangent modulus.
       df_dsigma = r.  There may be a lack of symmetry
       so we compute the full tensor (with lots of 
       inefficiencies) */ 
void 
YieldCondition::computeElasPlasTangentModulus(const Matrix3& r, 
                                              const Matrix3& df_ds, 
                                              const Matrix3& h_beta,
                                              const Matrix3& df_dbeta, 
                                              const double& h_alpha,             
                                              const double& df_dep,
                                              const double& h_phi,             
                                              const double& df_phi,
                                              const double& J,
                                              const double& dp_dJ,
                                              const PlasticityState* state,
                                              TangentModulusTensor& C_ep)
{
  Matrix3 one; one.Identity();

  // Compute terms in the denominator of B_ep
  double mu = state->shearModulus;
  Matrix3 dev_r = r - one*(r.Trace()/3.0);
  Matrix3 dev_h_beta = h_beta - one*(h_beta.Trace()/3.0);
  double term1 = (2.0*mu)*df_ds.Contract(dev_r);
  double term2 = df_dbeta.Contract(dev_h_beta);
  double term3 = df_dep*h_alpha;
  double term4 = df_phi*h_phi;
  double denom = term1 - (term2 + term3 + term4);
  double fac = (4.0*mu*mu)/denom;

  // Compute B_ep, i.e., Compute numerator/denom and subtract from 2*mu*I_4s
  TangentModulusTensor B_ep;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          double r_dfds = dev_r(ii,jj)*df_ds(kk,ll);
          double I_4s = 0.5*(one(ii,kk)*one(jj,ll)+one(ii,ll)*one(jj,kk));
          B_ep(ii,jj,kk,ll) = 2.0*mu*I_4s - fac*r_dfds;
        }
      }
    }
  }

  // Compute pressure factor
  double p_fac = J*dp_dJ;

  // Compute C_ep
  Matrix3 p_term_2(0.0);
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int mm = 0; mm < 3; ++mm) {
        for (int nn = 0; nn < 3; ++nn) {
          p_term_2(ii,jj) += B_ep(ii,jj,mm,nn)*one(mm,nn);
        }
      }
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          double p_term_1 = p_fac*(one(ii,jj)*one(kk,ll));
          double p_term_2_upd = p_term_2(ii,jj)*one(kk,ll)/3.0;
          C_ep(ii,jj,kk,ll) = p_term_1 - p_term_2_upd + B_ep(ii,jj,kk,ll);
        }
      }
    }
  }

  return;
}
