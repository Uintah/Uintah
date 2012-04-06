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


#include "ShearModulus_Borja.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace UintahBB;
using namespace std;

// Construct a shear modulus model.  
ShearModulus_Borja::ShearModulus_Borja(ProblemSpecP& ps )
{
  ps->require("mu0",d_mu0);
  ps->require("alpha",d_alpha);
  ps->require("p0",d_p0);
  ps->require("epse_v0",d_epse_v0);
  ps->require("kappatilde",d_kappatilde);
}

// Construct a copy of a shear modulus model.  
ShearModulus_Borja::ShearModulus_Borja(const ShearModulus_Borja* smm)
{
  d_mu0 = smm->d_mu0;
  d_alpha = smm->d_alpha;
  d_p0 = smm->d_p0;
  d_epse_v0 = smm->d_epse_v0;
  d_kappatilde = smm->d_kappatilde;
}

// Destructor of shear modulus model.  
ShearModulus_Borja::~ShearModulus_Borja()
{
}

void ShearModulus_Borja::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("elastic_shear_modulus_model");
  shear_ps->setAttribute("type","borja_shear_modulus");

  shear_ps->appendElement("mu_0",d_mu0);
  shear_ps->appendElement("alpha",d_alpha);
  shear_ps->appendElement("p0",d_p0);
  shear_ps->appendElement("epse_v0",d_epse_v0);
  shear_ps->appendElement("kappatilde",d_kappatilde);
}

         
// Compute the shear modulus
double 
ShearModulus_Borja::computeInitialShearModulus()
{
  double mu_vol = evalShearModulus(0.0);
  return (d_mu0 - mu_vol);
}

double 
ShearModulus_Borja::computeShearModulus(const ModelState* state) 
{
  double mu_vol = evalShearModulus(state->epse_v);
  return (d_mu0 - mu_vol);
}

double 
ShearModulus_Borja::computeShearModulus(const ModelState* state) const
{
  double mu_vol = evalShearModulus(state->epse_v);
  return (d_mu0 - mu_vol);
}

// Compute the shear strain energy
// W = 3/2 mu epse_s^2
double
ShearModulus_Borja::computeStrainEnergy(const ModelState* state)
{
  double mu_vol = evalShearModulus(state->epse_v);
  double W = 1.5*(d_mu0 - mu_vol)*(state->epse_s*state->epse_s);
  return W;
}

/* Compute q = 3 mu epse_s
         where mu = shear modulus
               epse_s = sqrt{2/3} ||ee||
               ee = deviatoric part of elastic strain = epse - 1/3 epse_v I
               epse = total elastic strain
               epse_v = tr(epse) */
double 
ShearModulus_Borja::computeQ(const ModelState* state) const
{
  return evalQ(state->epse_v, state->epse_s);
}

/* Compute dq/depse_s */
double 
ShearModulus_Borja::computeDqDepse_s(const ModelState* state) const
{
  return evalDqDepse_s(state->epse_v, state->epse_s);
}

/* Compute dq/depse_v */
double 
ShearModulus_Borja::computeDqDepse_v(const ModelState* state) const
{
  return evalDqDepse_v(state->epse_v, state->epse_s);
}

// Private methods below:

//  Shear modulus computation (only pressure contribution)
double 
ShearModulus_Borja::evalShearModulus(const double& epse_v) const
{
  double mu_vol = d_alpha*d_p0*exp(-(epse_v - d_epse_v0)/d_kappatilde);
  return mu_vol;
}

//  Shear stress magnitude computation
double 
ShearModulus_Borja::evalQ(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double q = 3.0*(d_mu0 - mu_vol)*epse_s;

  return q;
}

//  volumetric derivative computation
double 
ShearModulus_Borja::evalDqDepse_v(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double dmu_depse_v = mu_vol/d_kappatilde;
  double dq_depse_v = 3.0*dmu_depse_v*epse_s;
  return dq_depse_v;
}

//  deviatoric derivative computation
double 
ShearModulus_Borja::evalDqDepse_s(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double dq_depse_s = 3.0*(d_mu0 - mu_vol);
  return dq_depse_s;
}

