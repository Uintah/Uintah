/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#include "MieGruneisenEOSEnergy.h"
#include <cmath>

using namespace Uintah;
using namespace SCIRun;

MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(ProblemSpecP& ps)
{
  ps->require("C_0",d_const.C_0);
  ps->require("Gamma_0",d_const.Gamma_0);
  ps->require("S_alpha",d_const.S_1);
  ps->getWithDefault("S_2",d_const.S_2,0.0);
  ps->getWithDefault("S_3",d_const.S_3,0.0);
} 
         
MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(const MieGruneisenEOSEnergy* cm)
{
  d_const.C_0 = cm->d_const.C_0;
  d_const.Gamma_0 = cm->d_const.Gamma_0;
  d_const.S_1 = cm->d_const.S_1;
  d_const.S_2 = cm->d_const.S_2;
  d_const.S_3 = cm->d_const.S_3;
} 
         
MieGruneisenEOSEnergy::~MieGruneisenEOSEnergy()
{
}
         
void MieGruneisenEOSEnergy::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","mie_gruneisen");

  eos_ps->appendElement("C_0",d_const.C_0);
  eos_ps->appendElement("Gamma_0",d_const.Gamma_0);
  eos_ps->appendElement("S_alpha",d_const.S_1);
  eos_ps->appendElement("S_2",d_const.S_2);
  eos_ps->appendElement("S_3",d_const.S_3);
}

//////////
// Calculate the pressure using the Mie-Gruneisen equation of state
double 
MieGruneisenEOSEnergy::computePressure(const MPMMaterial* matl,
                                 const PlasticityState* state,
                                 const Matrix3& ,
                                 const Matrix3& ,
                                 const double& )
{
  // Get the current density
  double rho = state->density;

  // Get original density
  double rho_0 = matl->getInitialDensity();
   
  // Calc. eta
  double eta = 1. - rho_0/rho;

  // Retrieve specific internal energy e
  double e = state->energy;

  // Calculate the pressure
  double denom = 
               (1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta)
              *(1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta);
  double p;
  p = rho_0*d_const.Gamma_0*e 
    + rho_0*(d_const.C_0*d_const.C_0)*eta*(1. - .5*d_const.Gamma_0*eta)/denom;

  return -p;
}


double 
MieGruneisenEOSEnergy::computeIsentropicTemperatureRate(const double T,
                                                        const double rho_0,
                                                        const double rho_cur,
                                                        const double Dtrace)
{
  double dTdt = -T*d_const.Gamma_0*rho_0*Dtrace/rho_cur;

  return dTdt;
}

double 
MieGruneisenEOSEnergy::eval_dp_dJ(const MPMMaterial* matl,
                            const double& detF, 
                            const PlasticityState* state)
{
  double rho_0 = matl->getInitialDensity();
  double C_0 = d_const.C_0;
  double S_1 = d_const.S_1;
  double S_2 = d_const.S_2;
  double S_3 = d_const.S_3;
  double Gamma_0 = d_const.Gamma_0;

  double J = detF;

  double eta = 1.0-J;
  double denom = (1.0 - S_1*eta - S_2*eta*eta - S_3*eta*eta*eta);
  double numer = -rho_0*C_0*C_0*((1.0 - Gamma_0*eta)*denom 
       + 2.0*eta*(1.0 - Gamma_0*eta/2.0)*(S_1 + 2.0*S_2*eta + 3.0*S_3*eta*eta));
  double denom3 = (denom*denom*denom);

  if (denom3 == 0.0) {
    cout << "rh0_0 = " << rho_0 << " J = " << J 
           << " numer = " << numer << endl;
    denom3 = 1.0e-5;
  }

  return (numer/denom);
}
