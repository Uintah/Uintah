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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include "Pressure_Borja.h"
#include <Core/Math/DEIntegrator.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>

using namespace Uintah;
using namespace UintahBB;
using namespace std;

Pressure_Borja::Pressure_Borja(ProblemSpecP& ps)
{
  ps->require("p0",d_p0);
  ps->require("alpha",d_alpha);
  ps->require("kappatilde",d_kappatilde);
  ps->require("epse_v0",d_epse_v0);
} 
         
Pressure_Borja::Pressure_Borja(const Pressure_Borja* cm)
{
  d_p0 = cm->d_p0; 
  d_alpha = cm->d_alpha;
  d_kappatilde = cm->d_kappatilde;
  d_epse_v0 = cm->d_epse_v0;
} 
         
Pressure_Borja::~Pressure_Borja()
{
}
         
void Pressure_Borja::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("pressure_model");
  eos_ps->setAttribute("type","borja_pressure");

  ps->appendElement("p0",d_p0);
  ps->appendElement("alpha",d_alpha);
  ps->appendElement("kappatilde",d_kappatilde);
  ps->appendElement("epse_v0",d_epse_v0);
}

//////////
// Calculate the pressure using the Borja pressure model
//  (look at the header file for the equation)
double 
Pressure_Borja::computePressure(const MPMMaterial* ,
                                const ModelState* state,
                                const Matrix3& ,
                                const Matrix3& ,
                                const double& )
{
  double p = evalPressure(state->epse_v, state->epse_s);
  return p;
}

// Calculate the derivative of p with respect to epse_v
//      where epse_v = tr(epse)
//            epse = total elastic strain 
//   dp/depse_v = p0 beta/kappatilde exp[(epse_v - epse_v0)/kappatilde]
//              = p/kappatilde
double 
Pressure_Borja::computeDpDepse_v(const ModelState* state) const
{
  double dp_depse_v = evalDpDepse_v(state->epse_v, state->epse_s);
  return dp_depse_v;
}

// Calculate the derivative of p with respect to epse_s
//      where epse_s = sqrt{2}{3} ||ee||
//            ee = epse - 1/3 tr(epse) I
//            epse = total elastic strain 
double 
Pressure_Borja::computeDpDepse_s(const ModelState* state) const
{
  return evalDpDepse_s(state->epse_v, state->epse_s);
}

// Compute the derivative of p with respect to J
/* Assume dp/dJ = dp/deps_v (true for inifintesimal strains)
   Then
   dp/dJ = p0 beta/kappatilde exp[(epse_v - epse_v0)/kappatilde]
         = p/kappatilde
*/
double 
Pressure_Borja::eval_dp_dJ(const MPMMaterial* ,
                           const double& , 
                           const ModelState* state)
{
  return computeDpDepse_v(state);
}

// Set the initial value of the bulk modulus
void
Pressure_Borja::setInitialBulkModulus()
{
  d_bulk = evalDpDepse_v(0.0, 0.0);
  return;
}

// Compute incremental bulk modulus
double 
Pressure_Borja::computeBulkModulus(const ModelState* state)
{
  double K = evalDpDepse_v(state->epse_v, 0.0);
  return K;
}

// Compute volumetric strain energy
//   The strain energy function for the Borja model has the form
//      U(epse_v) = p0 kappatilde exp[(epse_v - epse_v0)/kappatilde]
double 
Pressure_Borja::computeStrainEnergy(const ModelState* state)
{
  double Wvol = -d_p0*d_kappatilde*exp(-(state->epse_v - d_epse_v0)/d_kappatilde);
  return Wvol;
}

// No isentropic increase in temperature with increasing strain
double 
Pressure_Borja::computeIsentropicTemperatureRate(const double ,
                                                     const double ,
                                                     const double ,
                                                     const double )
{
  return 0.0;
}

//--------------------------------------------------------------------------
// The following are needed for MPMICE coupling
//--------------------------------------------------------------------------

// Compute pressure (option 1) for MPMICE coupling
//   Assume epse_s = 0 for coupling purposes until the interface can be made
//   more general.
double 
Pressure_Borja::computePressure(const double& rho_orig,
                                    const double& rho_cur)
{
  // Calculate epse_v
  double epse_v = rho_orig/rho_cur - 1.0;
  double p = evalPressure(epse_v, 0.0);
  return p;
}

// Compute pressure (option 2) - for MPMICE coupling
//   Assume epse_s = 0 for coupling purposes until the interface can be made
//   more general.
//   c^2 = K/rho
//   dp/drho = -(J/rho) dp/depse_v = -(J/rho) K = -J c^2
void 
Pressure_Borja::computePressure(const double& rho_orig,
                                    const double& rho_cur,
                                    double& pressure,
                                    double& dp_drho,
                                    double& csquared)
{
  // Calculate J and epse_v
  double J = rho_orig/rho_cur;
  double epse_v = J - 1.0;

  pressure = evalPressure(epse_v, 0.0);
  double K = computeBulkModulus(rho_orig, rho_cur);
  csquared = K/rho_cur;
  dp_drho = - J*csquared;
  return;
}

// Compute the incremental bulk modulus
// The bulk modulus is defined at the tangent to the p - epse_v curve
// keeping epse_s fixed
// i.e., K = dp/depse_v 
// For the purposes of coupling to MPMICE we assume that epse_s = 0
// and epse_v = J - 1
double 
Pressure_Borja::computeInitialBulkModulus()
{
  double K = evalDpDepse_v(0.0, 0.0);
  return K;
}

double 
Pressure_Borja::computeBulkModulus(const double& rho_orig,
                                       const double& rho_cur)
{
  // Calculate epse_v
  double epse_v = rho_orig/rho_cur - 1.0;
  double K = evalDpDepse_v(epse_v, 0.0);
  return K;
}

// Compute density given pressure (tension +ve)
//  rho = rho0/[1 + epse_v0 - kappatilde ln(p/p0 beta)]
//  Assume epse_s = 0, i.e., beta = 1
double 
Pressure_Borja::computeDensity(const double& rho_orig,
                               const double& pressure)
{
  if (pressure > 0.0) return rho_orig;
  double denom = 1.0 + d_epse_v0 - d_kappatilde*log(pressure/d_p0);
  double rho = rho_orig/denom;
  return rho;
}

// Compute volumetric strain energy
//   The strain energy function for the Borja model has the form
//      U(epse_v) = p0 kappatilde exp[(epse_v - epse_v0)/kappatilde]
double 
Pressure_Borja::computeStrainEnergy(const double& rho_orig,
                                    const double& rho_cur)
{
  // Calculate epse_v
  double epse_v = rho_orig/rho_cur - 1.0;
  double Wvol = -d_p0*d_kappatilde*exp(-(epse_v - d_epse_v0)/d_kappatilde);
  return Wvol;
}

//-------------------------------------------------------------------------
// Private methods below:
//-------------------------------------------------------------------------

//  Pressure computation
double 
Pressure_Borja::evalPressure(const double& epse_v, const double& epse_s) const
{
  double beta = 1.0 + 1.5*(d_alpha/d_kappatilde)*(epse_s*epse_s);
  double p = d_p0*beta*exp(-(epse_v - d_epse_v0)/d_kappatilde);
  // std::cout << "beta = " << beta << " epse_v = " << epse_v << " p = " << p << endl;

  return p;
}

//  Pressure derivative computation
double 
Pressure_Borja::evalDpDepse_v(const double& epse_v, const double& epse_s) const
{
  double p = evalPressure(epse_v, epse_s);
  return -p/d_kappatilde;
}

//  Shear derivative computation
double 
Pressure_Borja::evalDpDepse_s(const double& epse_v, const double& epse_s) const
{
  double dbetaDepse_s = 3.0*(d_alpha/d_kappatilde)*epse_s;
  return d_p0*dbetaDepse_s*exp(-(epse_v - d_epse_v0)/d_kappatilde);
}


