
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MieGruneisenEOS.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

MieGruneisenEOS::MieGruneisenEOS(ProblemSpecP& ps)
{
  ps->require("C_0",d_const.C_0);
  ps->require("Gamma_0",d_const.Gamma_0);
  ps->require("S_alpha",d_const.S_alpha);
} 
	 
MieGruneisenEOS::~MieGruneisenEOS()
{
}
	 

//////////
// Calculate the pressure using the Mie-Gruneisen equation of state
Matrix3 
MieGruneisenEOS::computePressure(const MPMMaterial* matl,
                                 const double& ,
                                 const double& ,
                                 const Matrix3& ,
                                 const Matrix3& ,
                                 const Matrix3& ,
                                 const double& T,
                                 const double& rho,
                                 const double& )
{

   // Get original density
   double rho_0 = matl->getInitialDensity();
   
   // Calc. zeta
   double zeta = (rho/rho_0 - 1.0);
   if (zeta == 0) {
      Matrix3 zero(0.0);
      return zero;
   }

   // Calculate internal energy E
   double E = (matl->getSpecificHeat())*T;
 
   // Calculate the pressure
   double numer = rho_0*(d_const.C_0*d_const.C_0)*(1.0/zeta+(1.0-0.5*d_const.Gamma_0));
   double denom = 1.0/zeta - (d_const.S_alpha-1.0);
   double p = numer/(denom*denom) + d_const.Gamma_0*E;
   Matrix3 one; one.Identity();
   return (one*(-p));
}
