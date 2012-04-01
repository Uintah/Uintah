/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include "YieldCond_CamClay.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <cmath>

using namespace Uintah;
using namespace UintahBB;
using namespace std;

YieldCond_CamClay::YieldCond_CamClay(Uintah::ProblemSpecP& ps)
{
  ps->require("M",d_M);
}
         
YieldCond_CamClay::YieldCond_CamClay(const YieldCond_CamClay* yc)
{
  d_M = yc->d_M; 
}
         
YieldCond_CamClay::~YieldCond_CamClay()
{
}

void YieldCond_CamClay::outputProblemSpec(Uintah::ProblemSpecP& ps)
{
  ProblemSpecP yield_ps = ps->appendChild("plastic_yield_condition");
  yield_ps->setAttribute("type","camclay");
  yield_ps->appendElement("M",d_M);

}
         
//--------------------------------------------------------------
// Evaluate yield condition (q = state->q
//                           p = state->p
//                           p_c = state->p_c)
//--------------------------------------------------------------
double 
YieldCond_CamClay::evalYieldCondition(const ModelState* state)
{
  double p = state->p;
  double q = state->q;
  double p_c = state->p_c;
  return q*q/(d_M*d_M) + p*(p - p_c);
}

//--------------------------------------------------------------
// Derivatives needed by return algorithms and Newton iterations

//--------------------------------------------------------------
// Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
//   df/dp = 2p - p_c
//--------------------------------------------------------------
double 
YieldCond_CamClay::computeVolStressDerivOfYieldFunction(const ModelState* state)
{
  return (2.0*state->p - state->p_c);
}

//--------------------------------------------------------------
// Compute df/dq  
//   df/dq = 2q/M^2
//--------------------------------------------------------------
double 
YieldCond_CamClay::computeDevStressDerivOfYieldFunction(const ModelState* state)
{
  return 2.0*state->q/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute d/depse_v(df/dp)
//   df/dp = 2p(epse_v, epse_s) - p_c(epse_v)
//   d/depse_v(df/dp) = 2dp/depse_v - dp_c/depse_v
//
// Requires:  Equation of state and internal variable
//--------------------------------------------------------------
double
YieldCond_CamClay::computeVolStrainDerivOfDfDp(const ModelState* state,
                                               const PressureModel* eos,
                                               const ShearModulusModel* ,
                                               const InternalVariableModel* intvar)
{
  double dpdepsev = eos->computeDpDepse_v(state);
  double dpcdepsev = intvar->computeVolStrainDerivOfInternalVariable(state);
  return 2.0*dpdepsev - dpcdepsev;
}

//--------------------------------------------------------------
// Compute d/depse_s(df/dp)
//   df/dp = 2p(epse_v, epse_s) - p_c(epse_v)
//   d/depse_s(df/dp) = 2dp/depse_s 
//
// Requires:  Equation of state 
//--------------------------------------------------------------
double
YieldCond_CamClay::computeDevStrainDerivOfDfDp(const ModelState* state,
                                                   const PressureModel* eos,
                                                   const ShearModulusModel* ,
                                                   const InternalVariableModel* )
{
  double dpdepses = eos->computeDpDepse_s(state);
  return 2.0*dpdepses;
}

//--------------------------------------------------------------
// Compute d/depse_v(df/dq)
//   df/dq = 2q(epse_v, epse_s)/M^2
//   d/depse_v(df/dq) = 2/M^2 dq/depse_v
//
// Requires:  Shear modulus model
//--------------------------------------------------------------
double
YieldCond_CamClay::computeVolStrainDerivOfDfDq(const ModelState* state,
                                               const PressureModel* ,
                                               const ShearModulusModel* shear,
                                               const InternalVariableModel* )
{
  double dqdepsev = shear->computeDqDepse_v(state);
  return (2.0*dqdepsev)/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute d/depse_s(df/dq)
//   df/dq = 2q(epse_v, epse_s)/M^2
//   d/depse_s(df/dq) = 2/M^2 dq/depse_s
//
// Requires:  Shear modulus model
//--------------------------------------------------------------
double
YieldCond_CamClay::computeDevStrainDerivOfDfDq(const ModelState* state,
                                                   const PressureModel* ,
                                                   const ShearModulusModel* shear,
                                                   const InternalVariableModel* )
{
  double dqdepses = shear->computeDqDepse_s(state);
  return (2.0*dqdepses)/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute df/depse_v
//   df/depse_v = df/dq dq/depse_v + df/dp dp/depse_v - p dp_c/depse_v
//
// Requires:  Equation of state, shear modulus model, internal variable model
//--------------------------------------------------------------
double
YieldCond_CamClay::computeVolStrainDerivOfYieldFunction(const ModelState* state,
                                                            const PressureModel* eos,
                                                            const ShearModulusModel* shear,
                                                            const InternalVariableModel* intvar)
{
  double dfdq = computeDevStressDerivOfYieldFunction(state);
  double dfdp = computeVolStressDerivOfYieldFunction(state);
  double dqdepsev = shear->computeDqDepse_v(state);
  double dpdepsev = eos->computeDpDepse_v(state);
  double dpcdepsev = intvar->computeVolStrainDerivOfInternalVariable(state);
  double dfdepsev = dfdq*dqdepsev + dfdp*dpdepsev - state->p*dpcdepsev;

  return dfdepsev;
}

//--------------------------------------------------------------
// Compute df/depse_s
//   df/depse_s = df/dq dq/depse_s + df/dp dp/depse_s 
//
// Requires:  Equation of state, shear modulus model
//--------------------------------------------------------------
double
YieldCond_CamClay::computeDevStrainDerivOfYieldFunction(const ModelState* state,
                                                            const PressureModel* eos,
                                                            const ShearModulusModel* shear,
                                                            const InternalVariableModel* )
{
  double dfdq = computeDevStressDerivOfYieldFunction(state);
  double dfdp = computeVolStressDerivOfYieldFunction(state);
  double dqdepses = shear->computeDqDepse_s(state);
  double dpdepses = eos->computeDpDepse_s(state);
  double dfdepses = dfdq*dqdepses + dfdp*dpdepses;

  return dfdepses;
}

//--------------------------------------------------------------
// Other yield condition functions

// Evaluate yield condition (s = deviatoric stress
//                           p = state->p
//                           p_c = state->p_c)
double 
YieldCond_CamClay::evalYieldCondition(const Uintah::Matrix3& ,
                                      const ModelState* state)
{
  double p = state->p;
  double q = state->q;
  double pc = state->p_c;
  double dummy = 0.0;
  return evalYieldCondition(p, q, pc, 0.0, dummy);
}

double 
YieldCond_CamClay::evalYieldCondition(const double p,
                                      const double q,
                                      const double p_c,
                                      const double,
                                      double& )
{
  return q*q/(d_M*d_M) + p*(p - p_c);
}

//--------------------------------------------------------------
// Other derivatives 

// Compute df/dsigma
//    df/dsigma = (2p - p_c)/3 I + sqrt(3/2) 2q/M^2 s/||s||
//              = 1/3 df/dp I + sqrt(3/2) df/dq s/||s||
//              = 1/3 df/dp I + df/ds
// where
//    s = sigma - 1/3 tr(sigma) I
void 
YieldCond_CamClay::evalDerivOfYieldFunction(const Uintah::Matrix3& sig,
                                                const double p_c,
                                                const double ,
                                                Uintah::Matrix3& derivative)
{
  Matrix3 One; One.Identity();
  double p = sig.Trace()/3.0;
  Matrix3 sigDev = sig - One*p;
  double df_dp = 2.0*p - p_c;
  Matrix3 df_ds(0.0);
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  derivative = One*(df_dp/3.0) + df_ds;
  return;
}

// Compute df/ds  where s = deviatoric stress
//    df/ds = sqrt(3/2) df/dq s/||s|| = sqrt(3/2) 2q/M^2 n
void 
YieldCond_CamClay::evalDevDerivOfYieldFunction(const Uintah::Matrix3& sigDev,
                                                   const double ,
                                                   const double ,
                                                   Uintah::Matrix3& derivative)
{
  double sigDevNorm = sigDev.Norm();
  Matrix3 n = sigDev/sigDevNorm;
  double q_scaled = 3.0*sigDevNorm;
  derivative = n*(q_scaled/d_M*d_M);
  return;
}

/*! Derivative with respect to the Cauchy stress (\f$\sigma \f$) */
//   p_c = state->p_c
void 
YieldCond_CamClay::eval_df_dsigma(const Matrix3& sig,
                                      const ModelState* state,
                                      Matrix3& df_dsigma)
{
  evalDerivOfYieldFunction(sig, state->p_c, 0.0, df_dsigma);
  return;
}

/*! Derivative with respect to the \f$xi\f$ where \f$\xi = s \f$  
    where \f$s\f$ is deviatoric part of Cauchy stress */
void 
YieldCond_CamClay::eval_df_dxi(const Matrix3& sigDev,
                                   const ModelState* ,
                                   Matrix3& df_ds)
{
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  return;
}

/* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
void 
YieldCond_CamClay::eval_df_ds_df_dbeta(const Matrix3& sigDev,
                                           const ModelState*,
                                           Matrix3& df_ds,
                                           Matrix3& df_dbeta)
{
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  Matrix3 zero(0.0);
  df_dbeta = zero; 
  return;
}

/*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$) */
double 
YieldCond_CamClay::eval_df_dep(const Matrix3& ,
                                   const double& dsigy_dep,
                                   const ModelState* )
{
  cout << "YieldCond_CamClay: eval_df_dep not implemented yet " << endl;
  return 0.0;
}

/*! Derivative with respect to the porosity (\f$\epsilon^p \f$) */
double 
YieldCond_CamClay::eval_df_dphi(const Matrix3& ,
                                    const ModelState* )
{
  cout << "YieldCond_CamClay: eval_df_dphi not implemented yet " << endl;
  return 0.0;
}

/*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
double 
YieldCond_CamClay::eval_h_alpha(const Matrix3& ,
                                    const ModelState* )
{
  cout << "YieldCond_CamClay: eval_h_alpha not implemented yet " << endl;
  return 1.0;
}

/*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
double 
YieldCond_CamClay::eval_h_phi(const Matrix3& ,
                                  const double& ,
                                  const ModelState* )
{
  cout << "YieldCond_CamClay: eval_h_phi not implemented yet " << endl;
  return 0.0;
}

//--------------------------------------------------------------
// Tangent moduli
void 
YieldCond_CamClay::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
                                                     const Matrix3& sigma, 
                                                     double sigY,
                                                     double dsigYdep,
                                                     double porosity,
                                                     double ,
                                                     TangentModulusTensor& Cep)
{
  cout << "YieldCond_CamClay: computeElasPlasTangentModulus not implemented yet " << endl;
  return;
}

void 
YieldCond_CamClay::computeTangentModulus(const TangentModulusTensor& Ce,
                                             const Matrix3& f_sigma, 
                                             double f_q1,
                                             double h_q1,
                                             TangentModulusTensor& Cep)
{
  cout << "YieldCond_CamClay: computeTangentModulus not implemented yet " << endl;
  return;
}


