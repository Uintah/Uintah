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


#include "DeformationState.h"
using namespace Uintah;

DeformationState::DeformationState()
{
  deltaT = 0.0;                           // t_n+1 - t_n
  velGrad = Matrix3(0.0);                 // l_{n+1} = grad(v)_{n+1}
  velGrad_old = Matrix3(0.0);             // l_n = grad(v)_{n}
  defGrad = Matrix3::Identity();          // F_{n+1}
  defGrad_old = Matrix3::Identity();      // F_n
  defGrad_inc = Matrix3::Identity();      // Delta F = exp[Delta t * 0.5(l_n + l_{n+1})]
  rotation = Matrix3::Identity();         // R  in F = RU
  stretch = Matrix3::Identity();          // U  in F = RU
  rateOfDef = Matrix3(0.0);               // d_{n+1} = 0.5*[l_{n+1} + l_{n+1}^T]
  spin = Matrix3(0.0);                    // w_{n+1} = 0.5*[l_{n+1} - l_{n+1}^T]
  J = 1.0;                                // J_{n+1} = det(F_{n+1})
  J_inc = 1.0;                            // Delta J = det(Delta F)
  strain = Matrix3(0.0);                  // eps = whatever measure is being used.
  eps_v = 0.0;                            // eps_v = trace(eps) = volumetric strain
  dev_strain = Matrix3(0.0);              // eps_dev = eps - 1/3 eps_vol 1
  eps_s = 0.0;                            // eps_s = sqrt{2/3} ||eps_dev||
}

DeformationState::DeformationState(const DeformationState& state)
{
  copy(state);
}

DeformationState::DeformationState(const DeformationState* state)
{
  copy(state);
}

DeformationState::~DeformationState()
{
}

DeformationState&
DeformationState::operator=(const DeformationState& state)
{
  if (this == &state) return *this;
  copy(state);
  return *this;
}

DeformationState*
DeformationState::operator=(const DeformationState* state)
{
  if (this == state) return this;
  copy(state);
  return this;
}

void
DeformationState::copy(const DeformationState& state)
{
  deltaT = state.deltaT;
  velGrad = state.velGrad;
  velGrad_old = state.velGrad_old;
  defGrad = state.defGrad;
  defGrad_old = state.defGrad_old;
  defGrad_inc = state.defGrad_inc;
  rotation = state.rotation;
  stretch = state.stretch;
  rateOfDef = state.rateOfDef;
  spin = state.spin;
  J = state.J;
  J_inc = state.J_inc;
  strain = state.strain;
  eps_v = state.eps_v;
  dev_strain = state.dev_strain;
  eps_s = state.eps_s;
}

void
DeformationState::copy(const DeformationState* state)
{
  deltaT = state->deltaT;
  velGrad = state->velGrad;
  velGrad_old = state->velGrad_old;
  defGrad = state->defGrad;
  defGrad_old = state->defGrad_old;
  defGrad_inc = state->defGrad_inc;
  rotation = state->rotation;
  stretch = state->stretch;
  rateOfDef = state->rateOfDef;
  spin = state->spin;
  J = state->J;
  J_inc = state->J_inc;
  strain = state->strain;
  eps_v = state->eps_v;
  dev_strain = state->dev_strain;
  eps_s = state->eps_s;
}

// Compute and store the various deformation measures of interest starting with
//   dF/dt = l.F
//   Assuming linear (affine) l over the timestep such that
//     l(t) = (1-theta) l(t_n) + theta l(t_n+1)
//   where
//     theta = (t - t_n)/(t_n+1 - t_n) = (t - t_n)/Delta t
//   and the initial condition
//     F(t_n) = F_n
//   we have
//     F(t_n+1) = exp[0.5 (l_n + l_n+1) Delta t].F_n =: F_inc.F_n
//   For small Delta t, a two term Taylor series expansion of the exponential is reasonably
//   accurate and, if we define l_mid := 0.5 (l_n + l_n+1),  we can write
//     F_inc =  exp [l_mid Delta t] = 1 + l_mid Delta t + 0.5 (l_mid.l_mid) (Delta t)^2 + ...
//   Since det(exp(A)) = exp(tr(A)) we can find J_inc using
//     J_inc = exp(tr(l_mid) Delta_t)
//   This value can be compared with det(F_inc) to see whether there is a large error in 
//   the series approximation.
//   The polar decomposition can be calculated using Rebecca Branon's algorithm.
//   Strains should be calculated according to the requirements of the constitutive model.
void 
DeformationState::update(const Matrix3& l_old, const Matrix3& l_new, 
                         const Matrix3& F_old, const double& delT)
{
  // Save the velocity gradients and deformation gradient
  deltaT = delT;
  velGrad_old = l_old;
  velGrad = l_new;
  defGrad_old = F_old;

  // Compute mid point
  Matrix3 l_mid = (l_old + l_new)*0.5;

  // Compute increment of deformation gradient and the new deformation gradient
  Matrix3 One;  One.Identity();
  defGrad_inc = One + l_mid*delT + (l_mid*l_mid)*(0.5*delT*delT);
  defGrad = defGrad_inc*F_old;

  // Compute determinants
  J = defGrad.Determinant();
  J_inc = exp(l_mid.Trace()*delT);
  double J_inc_check = defGrad_inc.Determinant();
  if (fabs(J_inc - J_inc_check) > 1.0e-16) {
    cerr << "Taylor series approximation of F_inc not accurate enough." << endl;
  }

  // Compute rate of deformation and spin
  rateOfDef = (l_new + l_new.Transpose())*0.5;
  spin = (l_new - l_new.Transpose())*0.5;

  // Compute rotation and stretch
  defGrad.polarDecompositionRMB(stretch, rotation);

  // Compute hypoelastic strain (default)
  // computeHypoelasticStrain();
}

void 
DeformationState::computeHypoelasticStrain()
{
  // Compute hypoelastic strain
  Matrix3 One;  One.Identity();
  strain = rateOfDef*delT;
  eps_v = strain.Trace();
  dev_strain = strain - One*(eps_v/3.0);
  eps_s = sqrt(2.0/3.0)*dev_strain.Norm();
}

// strain = 0.5 (Ft.F - 1)
void 
DeformationState::computeGreenStrain()
{
  // Compute Green strain
  Matrix3 One;  One.Identity();
  strain = (defGrad.Transpose()*defGrad - One)*0.5;
  eps_v = strain.Trace();
  dev_strain = strain - One*(eps_v/3.0);
  eps_s = sqrt(2.0/3.0)*dev_strain.Norm();
}

// strain = 0.5 (1 - Finv_t.Finv)
void 
DeformationState::computeAlmansiStrain()
{
  // Compute Green strain
  Matrix3 One;  One.Identity();
  Matrix3 Finv = defGrad.Inverse();
  strain = (One - Finv.Transpose()*Finv)*0.5;
  eps_v = strain.Trace();
  dev_strain = strain - One*(eps_v/3.0);
  eps_s = sqrt(2.0/3.0)*dev_strain.Norm();
}

// strain = B =  F.Ft
void 
DeformationState::computeCauchyGreenB()
{
  // Compute left Cauchy-Green deformation tensor
  Matrix3 One;  One.Identity();
  double twoThird = 2.0/3.0;
  strain = defGrad*defGrad.Transpose();
  eps_v = strain.Trace();
  dev_strain = strain - One*(eps_v/3.0);
  eps_s = sqrt(twoThird)*dev_strain.Norm();
}

// strain = B_bar = J^(-2/3) B = J^{-2/3) F.Ft
void 
DeformationState::computeCauchyGreenBbar()
{
  // Compute left Cauchy-Green deformation tensor
  Matrix3 One;  One.Identity();
  double twoThird = 2.0/3.0;
  strain = defGrad*defGrad.Transpose()*pow(1.0/J, twoThird);
  eps_v = strain.Trace();
  dev_strain = strain - One*(eps_v/3.0);
  eps_s = sqrt(twoThird)*dev_strain.Norm();
}
