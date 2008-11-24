
#include "MieGruneisenEOSEnergy.h"
#include <cmath>

using namespace Uintah;
using namespace SCIRun;

MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(ProblemSpecP& ps)
{
  ps->require("C_0",d_const.C_0);
  ps->require("Gamma_0",d_const.Gamma_0);
  ps->require("S_alpha",d_const.S_alpha);
} 
	 
MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(const MieGruneisenEOSEnergy* cm)
{
  d_const.C_0 = cm->d_const.C_0;
  d_const.Gamma_0 = cm->d_const.Gamma_0;
  d_const.S_alpha = cm->d_const.S_alpha;
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
  eos_ps->appendElement("S_alpha",d_const.S_alpha);
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
  double denom = (1.-d_const.S_alpha*eta)*(1.-d_const.S_alpha*eta);
  double p;
  p = rho_0*d_const.Gamma_0*e 
    + rho_0*(d_const.C_0*d_const.C_0)*eta*(1. - .5*d_const.Gamma_0*eta)/denom;

  return -p;
}


double 
MieGruneisenEOSEnergy::computeIsentropicTemperatureIncrement(const double T,
                                                       const double rho_0,
                                                       const double rho_cur,
                                                       const double Dtrace,
                                                       const double delT)
{
  double dT = -T*d_const.Gamma_0*rho_0*Dtrace*delT/rho_cur;

  return dT;
}

double 
MieGruneisenEOSEnergy::eval_dp_dJ(const MPMMaterial* matl,
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
