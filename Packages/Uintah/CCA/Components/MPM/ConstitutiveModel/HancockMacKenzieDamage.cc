
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HancockMacKenzieDamage.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

HancockMacKenzieDamage::HancockMacKenzieDamage(ProblemSpecP& ps)
{
  d_initialData.D0 = 0.0;
  ps->get("D0",d_initialData.D0);
  ps->require("Dc",d_initialData.Dc);
} 
	 
HancockMacKenzieDamage::~HancockMacKenzieDamage()
{
}
	 
inline double 
HancockMacKenzieDamage::initialize()
{
  return d_initialData.D0;
}

inline bool
HancockMacKenzieDamage:: hasFailed(double damage)
{
  if (damage > d_initialData.Dc) return true;
  return false;
}
    
double 
HancockMacKenzieDamage::computeScalarDamage(const Matrix3& rateOfDeformation,
					    const Matrix3& stress,
					    const double& temperature,
					    const double& delT,
					    const MPMMaterial* matl,
					    const double& tolerance,
					    const double& D_old)
{
  // Calculate plastic strain rate
  double plasticStrainRate = sqrt(rateOfDeformation.NormSquared()*2.0/3.0);
  double epsInc = plasticStrainRate*delT;

  // Compute hydrostatic stress and equivalent stress
  double sig_h = stress.Trace()/3.0;
  double sig_eq = sqrt((stress.NormSquared())*1.5);

  // Calculate the updated scalar damage parameter
  double D = D_old + (1.0/1.65)*epsInc*exp(1.5*sig_h/sig_eq);
  return D;
}
 
