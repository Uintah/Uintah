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

#include "DefaultHyperelasticPressure.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct the default pressure model for hyperelastic materials.  
//            W = D1 (J-1)^2 = 0.5*kappa*(J-1)^2
//            p = dW/dJ = kappa*(J-1)
//            dp/dJ = d2W/dJ^2 = kappa
//            dp/drho = dp/dJ dJ/drho = - kappa J/rho
//            kappa_tan = p + J dp/dJ = kappa*(2J-1)
//            c^2 = k/rho
//            rho = rho0*kappa/(p + kappa)
DefaultHyperelasticPressure::DefaultHyperelasticPressure(ProblemSpecP& ps )
{
  ps->require("bulk_modulus", d_kappa);
}

// Construct a copy of a pressure model.  
DefaultHyperelasticPressure::DefaultHyperelasticPressure(const DefaultHyperelasticPressure* pm)
{
  d_kappa = pm->d_kappa;
}

// Destructor of pressure model.  
DefaultHyperelasticPressure::~DefaultHyperelasticPressure()
{
}

void DefaultHyperelasticPressure::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP pressure_ps = ps->appendChild("pressure_model");
  pressure_ps->setAttribute("type", "hyperelastic");
  pressure_ps->appendElement("bulk_modulus", d_kappa);
}

// Compute the total pressure 
//   p = kappa (J-1)
double 
DefaultHyperelasticPressure::computePressure(const DeformationState* state)
{
  return d_kappa*(state->J - 1.0);
}

// Calculate the pressure  (option 1)
double 
DefaultHyperelasticPressure::computePressure(const double& rho_orig,
                                             const double& rho_cur)
{
   return d_kappa*(rho_orig/rho_cur - 1.0);
}

// Calculate the pressure without considering internal energy (option 2).  
//   Also compute dp/drho and c^2. 
//   c^2 = kappa_cur/rho_cur
void 
DefaultHyperelasticPressure::computePressure(const double& rho_orig,
                                             const double& rho_cur,
                                             double& pressure,
                                             double& dp_drho,
                                             double& csquared)
{
   double J = rho_orig/rho_cur;
   pressure = d_kappa*(J - 1.0);
   dp_drho = -d_kappa*J/rho_cur;
   double kappa_cur = d_kappa*(2.0*J - 1.0);
   csquared = kappa_cur/rho_cur;
}

// Calculate the tangent bulk modulus 
double 
DefaultHyperelasticPressure::computeTangentBulkModulus(const double& rho_orig,
                                                      const double& rho_cur)
{
   return d_kappa*(2.0*rho_orig/rho_cur - 1.0);
}

// Calculate the accumulated strain energy 
//   w = 0.5*kappa*(J-1)^2
double 
DefaultHyperelasticPressure::computeStrainEnergy(const double& pressure,
                                                 const DeformationState* state)
{
   double Jone = rho_orig/rho_cur - 1.0;
   return 0.5*d_kappa*Jone*Jone;
}

// Calculate the mass density at a given pressure 
//   rho = rho0*kappa/(p + kappa)
double 
DefaultHyperelasticPressure::computeDensity(const double& rho_orig,
                                            const double& pressure)
{
    return rho_orig*d_kappa/(pressure + d_kappa);
}
