
#include "JohnsonCookDamage.h"
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
  d_initialData.spallStress = 8.0;
} 
	 
JohnsonCookDamage::JohnsonCookDamage(const JohnsonCookDamage* cm)
{
  d_initialData.D1 = cm->d_initialData.D1;
  d_initialData.D2 = cm->d_initialData.D2;
  d_initialData.D3 = cm->d_initialData.D3;
  d_initialData.D4 = cm->d_initialData.D4;
  d_initialData.D5 = cm->d_initialData.D5;
  d_initialData.D0 = cm->d_initialData.D0;
  d_initialData.Dc = cm->d_initialData.Dc;
  d_initialData.spallStress = cm->d_initialData.spallStress;
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
JohnsonCookDamage::computeScalarDamage(const double& epdot,
                                       const Matrix3& stress,
                                       const double& T,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const double& tolerance,
                                       const double& damage_old)
{
  Matrix3 I; I.Identity();
  double sigMean = stress.Trace()/3.0;
  Matrix3 sig_dev = stress - I*sigMean;
  double sigEquiv = sqrt((sig_dev.NormSquared())*1.5);
  //cout << "sigMean = " << sigMean << " sigEquiv = " << sigEquiv;

  double sigStar = 0.0;
  if (sigEquiv != 0) sigStar = sigMean/sigEquiv;
  //if (sigStar > d_initialData.spallStress) return 1.0;
  if (sigStar > 1.5) sigStar = 1.5;
  if (sigStar < -1.5) sigStar = -1.5;
  //cout << " sigStar = " << sigStar;
  double stressPart = d_initialData.D1 + 
    d_initialData.D2*exp(d_initialData.D3*sigStar);
  //cout << " stressPart = " << stressPart;

  double strainRatePart = 1.0;
  //cout << " epdot = " << epdot;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.D4);
  else
    strainRatePart = 1.0 + d_initialData.D4*log(epdot);
  //cout << " epdotPart = " << strainRatePart;

  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  //cout << " Tr = " << Tr << " Tm = " << Tm << " T = " << T << endl;
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = 1.0 + d_initialData.D5*Tstar;
  //cout << " tempPart = " << tempPart;

  // Calculate the updated scalar damage parameter
  double epsFrac = stressPart*strainRatePart*tempPart;
  if (epsFrac < tolerance) return damage_old;

  // Calculate plastic strain increment
  double epsInc = epdot*delT;
  double damage_new = damage_old  + epsInc/epsFrac;
  if (damage_new < tolerance) damage_new = 0.0;
  /*
  cout << "sigstar = " << sigStar << " epdotStar = " << epdot
       << " Tstar = " << Tstar << endl;
  cout << "Ep_dot = " << epdot 
       << " e_inc = " << epsInc
       << " e_f = " << epsFrac
       << " D_n = " << damage_old 
       << " D_n+1 = " << damage_new << endl;
  */
  return damage_new;
}
 
