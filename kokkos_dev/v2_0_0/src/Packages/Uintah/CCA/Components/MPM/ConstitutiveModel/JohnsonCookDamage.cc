
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookDamage.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

JohnsonCookDamage::JohnsonCookDamage(ProblemSpecP& ps)
{
  ps->require("D1",d_initialData.D1);
  ps->require("D2",d_initialData.D2);
  ps->require("D3",d_initialData.D3);
  ps->require("D4",d_initialData.D4);
  ps->require("D5",d_initialData.D5);
  d_initialData.D0 = 0.0;
  ps->get("D0",d_initialData.D0);
  d_initialData.Dc = 0.7;
  ps->get("Dc",d_initialData.Dc);
} 
	 
JohnsonCookDamage::~JohnsonCookDamage()
{
}
	 
inline double 
JohnsonCookDamage::initialize()
{
  return d_initialData.D0;
}

inline bool
JohnsonCookDamage:: hasFailed(double damage)
{
  if (damage > d_initialData.Dc) return true;
  return false;
}
    
double 
JohnsonCookDamage::computeScalarDamage(const Matrix3& rateOfDeformation,
                                       const Matrix3& stress,
                                       const double& temperature,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const double& tolerance,
                                       const double& damage_old)
{
  // Calculate plastic strain rate
  double plasticStrainRate = sqrt(rateOfDeformation.NormSquared()*2.0/3.0);
  double epsInc = plasticStrainRate*delT;

  // Calculate the updated scalar damage parameter
  double epsFrac = calcStrainAtFracture(stress, plasticStrainRate, temperature,
                                        matl, tolerance);
  //cout << "Plastic Strain rate = " << plasticStrainRate 
  //     << "Plastic Strain Increment = " << epsInc
  //     << "Strain At fracture = " << epsFrac
  //     << "Damage_old = " << damage_old << endl;

  if (epsFrac == 0) return damage_old;
  return (damage_old  + epsInc/epsFrac);
}
 
double 
JohnsonCookDamage::calcStrainAtFracture(const Matrix3& stress, 
                                        const double& epdot,
                                        const double& T,
                                        const MPMMaterial* matl,
                                        const double& )
{
  double sigMean = stress.Trace()/3.0;
  double sigEquiv = sqrt((stress.NormSquared())*1.5);
  double sigStar = 0.0;
  if (sigEquiv != 0) sigStar = sigMean/sigEquiv;
  double stressPart = d_initialData.D1 + 
    d_initialData.D2*exp(d_initialData.D3*sigStar);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.D4);
  else
    strainRatePart = 1.0 + d_initialData.D4*log(epdot);
  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = 1.0 + d_initialData.D5*Tstar;
  return (stressPart*strainRatePart*tempPart);
}
