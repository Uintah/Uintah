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


#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include "LinearElasticPressure.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a pressure model.  
LinearElasticPressure::LinearElasticPressure(ProblemSpecP& ps )
{
  ps->require("bulk_modulus", d_kappa);
}

// Construct a copy of a pressure model.  
LinearElasticPressure::LinearElasticPressure(const LinearElasticPressure* pm)
{
  d_kappa = pm->d_kappa;
}

// Destructor of pressure model.  
LinearElasticPressure::~LinearElasticPressure()
{
}

void LinearElasticPressure::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP pressure_ps = ps->appendChild("pressure_model");
  pressure_ps->setAttribute("type", "linear_elastic");
  pressure_ps->appendElement("bulk_modulus", d_kappa);
}

// Compute the pressure (increment in this case)
//   Delta p = kappa*tr(Delta t * rate_of_deformation)
double 
LinearElasticPressure::computePressure(const DeformationState* state)
{
  state->computeHypoelasticStrain();
  return state->eps_v*d_kappa;
}

// Calculate the pressure without considering internal energy (option 1)
//   p = kappa Tr(eps) = kappa (V-V0)/V0 = kappa (rho0/rho - 1)
double 
LinearElasticPressure::computePressure(const double& rho_orig,
                                       const double& rho_cur)
{
   return d_kappa*(rho_orig/rho_cur - 1);
}

// Calculate the pressure without considering internal energy (option 2).  
//   Also compute dp/drho and c^2. 
//   p = kappa Tr(eps) = kappa (V-V0)/V0 = kappa (rho0/rho - 1)
//   dp_drho = -kappa rho0/rho^2
//   c^2 = kappa/rho
void 
LinearElasticPressure::computePressure(const double& rho_orig,
                                       const double& rho_cur,
                                       double& pressure,
                                       double& dp_drho,
                                       double& csquared)
{
   pressure = d_kappa*(rho_orig/rho_cur - 1);
   dp_drho = -d_kappa*rho_orig/(rho_cur*rho_cur);
   csquared = d_kappa/rho_cur;
}

// Calculate the tangent bulk modulus 
double 
LinearElasticPressure::computeTangentBulkModulus(const double& rho_orig,
                                                 const double& rho_cur)
{
   return d_kappa;
}

// Calculate the accumulated strain energy (increment)
//   Delta W_pressure = p*Tr(d)*Delta t
double 
LinearElasticPressure::computeStrainEnergy(const double& pressure,
                                           const DeformationState* state)
{
   return pressure*eps_v; 
}

// Calculate the mass density at a given pressure 
//   rho = rho0*kappa/(p + kappa)
double 
LinearElasticPressure::computeDensity(const double& rho_orig,
                                      const double& pressure)
{
    return rho_orig*d_kappa/(pressure + d_kappa);
}
