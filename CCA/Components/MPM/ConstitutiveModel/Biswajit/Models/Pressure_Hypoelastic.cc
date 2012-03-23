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



#include "Pressure_Hypoelastic.h"
#include <Core/Exceptions/ParameterNotFound.h>
#include <cmath>

using namespace Uintah;

Pressure_Hypoelastic::Pressure_Hypoelastic()
{
  d_bulk = -1.0;
} 

Pressure_Hypoelastic::Pressure_Hypoelastic(ProblemSpecP& )
{
  d_bulk = -1.0;
} 
         
Pressure_Hypoelastic::Pressure_Hypoelastic(const Pressure_Hypoelastic* cm)
{
  d_bulk = cm->d_bulk;
} 
         
Pressure_Hypoelastic::~Pressure_Hypoelastic()
{
}


void Pressure_Hypoelastic::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","default_hypo");
}
         

//////////
// Calculate the pressure using the elastic constitutive equation
double 
Pressure_Hypoelastic::computePressure(const MPMMaterial* ,
                                       const ModelState* state,
                                       const Matrix3& ,
                                       const Matrix3& rateOfDeformation,
                                       const double& delT)
{
  // Get the state data
  double kappa = state->bulkModulus;
  double p_n = state->pressure;

  // Calculate pressure increment
  double delp = rateOfDeformation.Trace()*(kappa*delT);

  // Calculate pressure
  double p = p_n + delp;
  return p;
}

double 
Pressure_Hypoelastic::eval_dp_dJ(const MPMMaterial* matl,
                                  const double& detF, 
                                  const ModelState* state)
{
  return (state->bulkModulus/detF);
}

// Compute pressure (option 1)  
//  (assume linear relation holds and bulk modulus does not change
//   with deformation)
double 
Pressure_Hypoelastic::computePressure(const double& rho_orig,
                                       const double& rho_cur)
{
  if (d_bulk < 0.0) {
    throw ParameterNotFound("Please initialize bulk modulus in EOS before computing pressure",
                            __FILE__, __LINE__);
  }

  double J = rho_orig/rho_cur;
  double p = d_bulk*(1.0 - 1.0/J);
  return p;
}

// Compute pressure (option 2)  (assume small strain relation holds)
//  (assume linear relation holds and bulk modulus does not change
//   with deformation)
void 
Pressure_Hypoelastic::computePressure(const double& rho_orig,
                                       const double& rho_cur,
                                       double& pressure,
                                       double& dp_drho,
                                       double& csquared)
{
  if (d_bulk < 0.0) {
    throw ParameterNotFound("Please initialize bulk modulus in EOS before computing pressure",
                            __FILE__, __LINE__);
  }

  double J = rho_orig/rho_cur;
  pressure = d_bulk*(1.0 - 1.0/J);
  dp_drho  = -d_bulk/rho_orig;
  csquared = d_bulk/rho_cur;
}

// Compute bulk modulus
double 
Pressure_Hypoelastic::computeInitialBulkModulus()
{
  return d_bulk;  // return -1 if uninitialized
}

double 
Pressure_Hypoelastic::computeBulkModulus(const double& rho_orig,
                                          const double& rho_cur)
{
  if (d_bulk < 0.0) {
    throw ParameterNotFound("Please initialize bulk modulus in EOS before computing modulus",
                            __FILE__, __LINE__);
  }

  double J = rho_orig/rho_cur;
  double bulk = d_bulk/(J*J);
  return bulk;
}

// Compute strain energy
// (integrate the equation for p)
double 
Pressure_Hypoelastic::computeStrainEnergy(const double& rho_orig,
                                           const double& rho_cur)
{
  if (d_bulk < 0.0) {
    throw ParameterNotFound("Please initialize bulk modulus in EOS before computing energy",
                            __FILE__, __LINE__);
  }

  double J = rho_orig/rho_cur;
  double U = d_bulk*(J - 1.0 - log(J));
  return U;
}

// Compute density given pressure
double 
Pressure_Hypoelastic::computeDensity(const double& rho_orig,
                                      const double& pressure)
{
  if (d_bulk < 0.0) {
    throw ParameterNotFound("Please initialize bulk modulus in EOS before computing density",
                            __FILE__, __LINE__);
  }

  double rho_cur = rho_orig*(1.0 - pressure/d_bulk);
  return rho_cur;
}
