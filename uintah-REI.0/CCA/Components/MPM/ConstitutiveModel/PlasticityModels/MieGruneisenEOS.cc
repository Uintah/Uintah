
#include "MieGruneisenEOS.h"
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

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
