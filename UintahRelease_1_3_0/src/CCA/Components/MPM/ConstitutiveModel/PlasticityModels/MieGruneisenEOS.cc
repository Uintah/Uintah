/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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



#include "MieGruneisenEOS.h"
#include <cmath>

using namespace Uintah;

MieGruneisenEOS::MieGruneisenEOS(ProblemSpecP& ps)
{
  ps->require("C_0",d_const.C_0);
  ps->require("Gamma_0",d_const.Gamma_0);
  ps->require("S_alpha",d_const.S_alpha);
} 
         
MieGruneisenEOS::MieGruneisenEOS(const MieGruneisenEOS* cm)
{
  d_const.C_0 = cm->d_const.C_0;
  d_const.Gamma_0 = cm->d_const.Gamma_0;
  d_const.S_alpha = cm->d_const.S_alpha;
} 
         
MieGruneisenEOS::~MieGruneisenEOS()
{
}
         
void MieGruneisenEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","mie_gruneisen");

  eos_ps->appendElement("C_0",d_const.C_0);
  eos_ps->appendElement("Gamma_0",d_const.Gamma_0);
  eos_ps->appendElement("S_alpha",d_const.S_alpha);
}

//////////
// Calculate the pressure using the Mie-Gruneisen equation of state
double 
MieGruneisenEOS::computePressure(const MPMMaterial* matl,
                                 const PlasticityState* state,
                                 const Matrix3& ,
                                 const Matrix3& ,
                                 const double& )
{
  // Get the state data
  double rho = state->density;
  double T = state->temperature;
  double T_0 = state->initialTemperature;

  // Get original density
  double rho_0 = matl->getInitialDensity();
   
  // Calc. zeta
  double zeta = (rho/rho_0 - 1.0);

  // Calculate internal energy E
  double E = (state->specificHeat)*(T - T_0)*rho_0;
 
  // Calculate the pressure
  double p = d_const.Gamma_0*E;
  if (rho != rho_0) {
    double numer = rho_0*(d_const.C_0*d_const.C_0)*(1.0/zeta+
                         (1.0-0.5*d_const.Gamma_0));
    double denom = 1.0/zeta - (d_const.S_alpha-1.0);
    if (denom == 0.0) {
      cout << "rh0_0 = " << rho_0 << " zeta = " << zeta 
           << " numer = " << numer << endl;
      denom = 1.0e-5;
    }
    p += numer/(denom*denom);
  }
  return -p;
}

double 
MieGruneisenEOS::eval_dp_dJ(const MPMMaterial* matl,
                            const double& detF, 
                            const PlasticityState* state)
{
  double rho_0 = matl->getInitialDensity();
  double C_0 = d_const.C_0;
  double S_alpha = d_const.S_alpha;
  double Gamma_0 = d_const.Gamma_0;

  double J = detF;
  double numer = rho_0*C_0*C_0*(1.0 + (S_alpha - Gamma_0)*(1.0-J));
  double denom = (1.0 - S_alpha*(1.0-J));
  double denom3 = (denom*denom*denom);
  if (denom3 == 0.0) {
    cout << "rh0_0 = " << rho_0 << " J = " << J 
           << " numer = " << numer << endl;
    denom3 = 1.0e-5;
  }

  return (numer/denom);
}
