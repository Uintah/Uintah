
#include "HancockMacKenzieDamage.h"
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

HancockMacKenzieDamage::HancockMacKenzieDamage(ProblemSpecP& ps)
{
  d_initialData.D0 = 0.0;
  ps->get("D0",d_initialData.D0);
  ps->require("Dc",d_initialData.Dc);
} 
         
HancockMacKenzieDamage::HancockMacKenzieDamage(const HancockMacKenzieDamage* cm)
{
  d_initialData.D0  = cm->d_initialData.D0;
  d_initialData.Dc  = cm->d_initialData.Dc;
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
HancockMacKenzieDamage::hasFailed(double damage)
{
  if (damage > d_initialData.Dc) return true;
  return false;
}
    
double 
HancockMacKenzieDamage::computeScalarDamage(const double& plasticStrainRate,
                                            const Matrix3& stress,
                                            const double& ,
                                            const double& delT,
                                            const MPMMaterial* ,
                                            const double& ,
                                            const double& D_old)
{
  // Calculate plastic strain increment
  double epsInc = plasticStrainRate*delT;

  // Compute hydrostatic stress and equivalent stress
  double sig_h = stress.Trace()/3.0;
  Matrix3 I; I.Identity();
  Matrix3 sig_dev = stress - I*sig_h;
  double sig_eq = sqrt((sig_dev.NormSquared())*1.5);

  // Calculate the updated scalar damage parameter
  double D = D_old + (1.0/1.65)*epsInc*exp(1.5*sig_h/sig_eq);
  return D;
}
 
