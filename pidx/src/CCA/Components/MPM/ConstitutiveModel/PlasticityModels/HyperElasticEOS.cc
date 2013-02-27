/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include "HyperElasticEOS.h"
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>

using namespace Uintah;

HyperElasticEOS::HyperElasticEOS()
{
  d_bulk = -1.0;
} 

HyperElasticEOS::HyperElasticEOS(ProblemSpecP&)
{
  d_bulk = -1.0;
} 
         
HyperElasticEOS::HyperElasticEOS(const HyperElasticEOS* cm)
{
  d_bulk = cm->d_bulk;
} 
         
HyperElasticEOS::~HyperElasticEOS()
{
}

void HyperElasticEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","default_hyper");
}
         

//////////
// Calculate the pressure using the elastic constitutive equation
double 
HyperElasticEOS::computePressure(const MPMMaterial* matl,
                                       const PlasticityState* state,
                                       const Matrix3& ,
                                       const Matrix3& rateOfDeformation,
                                       const double& delT)
{
  double rho_0 = matl->getInitialDensity();
  double rho = state->density;
  double J = rho_0/rho;
  double kappa = state->bulkModulus;

  double p = 0.5*kappa*(J - 1.0/J);
  return p;
}

double 
HyperElasticEOS::eval_dp_dJ(const MPMMaterial* matl,
                                  const double& detF, 
                                  const PlasticityState* state)
{
  double J = detF;
  double kappa = state->bulkModulus;

  double dpdJ = 0.5*kappa*(1.0 + 1.0/(J*J));
  return dpdJ;
}

// Compute pressure (option 1)
double 
HyperElasticEOS::computePressure(const double& rho_orig,
                                 const double& rho_cur)
{
  /*
  if (d_bulk < 0.0) {
    throw InternalError("Please initialize bulk modulus in EOS before computing pressure",
                            __FILE__, __LINE__);
  }
  */

  double J = rho_orig/rho_cur;
  double p = 0.5*d_bulk*(J - 1.0/J);
  return p;
}

// Compute pressure (option 2)
void 
HyperElasticEOS::computePressure(const double& rho_orig,
                                 const double& rho_cur,
                                 double& pressure,
                                 double& dp_drho,
                                 double& csquared)
{
  /*
  if (d_bulk < 0.0) {
    throw InternalError("Please initialize bulk modulus in EOS before computing pressure",
                            __FILE__, __LINE__);
  }
  */

  double J = rho_orig/rho_cur;
  pressure = 0.5*d_bulk*(J - 1.0/J);
  double dp_dJ = 0.5*d_bulk*(1.0 + 1.0/J*J);
  dp_drho = -0.5*d_bulk*(1.0 + J*J)/rho_orig;
  csquared = dp_dJ/rho_cur;
}

// Compute bulk modulus
double 
HyperElasticEOS::computeBulkModulus(const double& rho_orig,
                                    const double& rho_cur)
{
  /*
  if (d_bulk < 0.0) {
    throw InternalError("Please initialize bulk modulus in EOS before computing modulus",
                            __FILE__, __LINE__);
  }
  */

  double J = rho_orig/rho_cur;
  double bulk = 0.5*d_bulk*(1.0 + 1.0/J*J);
  return bulk;
}

// Compute strain energy
double 
HyperElasticEOS::computeStrainEnergy(const double& rho_orig,
                                     const double& rho_cur)
{
  /*
  if (d_bulk < 0.0) {
    throw InternalError("Please initialize bulk modulus in EOS before computing energy",
                            __FILE__, __LINE__);
  }
  */

  double J = rho_orig/rho_cur;
  double U = 0.5*d_bulk*(0.5*(J*J - 1.0) - log(J));
  return U;
}

// Compute density given pressure (tension +ve)
double 
HyperElasticEOS::computeDensity(const double& rho_orig,
                                const double& pressure)
{
  /*
  if (d_bulk < 0.0) {
    throw InternalError("Please initialize bulk modulus in EOS before computing density",
                            __FILE__, __LINE__);
  }
  */
  double numer1 = d_bulk*d_bulk + pressure*pressure;
  double sqrtNumer = sqrt(numer1);
  double rho = rho_orig/d_bulk*(-pressure + sqrtNumer);
  if (rho < 0) {
    ostringstream desc;
    desc << "Value of pressure (" << pressure << ") is beyond the range of validity of model" << endl
         << "  density = " << rho << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
  return rho;
}
