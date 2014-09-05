
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

ConstitutiveModel::ConstitutiveModel()
{
   // Constructor
  lb = scinew MPMLabel();

}

ConstitutiveModel::~ConstitutiveModel()
{
  delete lb;
}

//______________________________________________________________________
//          HARDWIRE FOR AN IDEAL GAS -Todd 
double ConstitutiveModel::computeRhoMicro(double& press, double& gamma,
				 double& cv, double& Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

void ConstitutiveModel::computePressEOS(double& rhoM, double& gamma,
			       double& cv, double& Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
//______________________________________________________________________
//
// The "CM" versions use the pressure-volume relationship of the CNH model
double ConstitutiveModel::computeRhoMicroCM(double pressure,
					  const MPMMaterial* matl)
{

  // This is really only valid now for CompNeoHook(Plas)
  double rho_orig = matl->getInitialDensity();
  double p_ref=101325.0;
  double bulk = 2000.0;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

//  cout << rho_cur << endl;

  return rho_cur;
}

void ConstitutiveModel::computePressEOSCM(const double rho_cur,double& pressure,
                                                double& dp_drho, double& tmp,
					        const MPMMaterial* matl)
{

  // This is really only valid now for CompNeoHook(Plas)
  double p_ref=101325.0;
  double bulk = 2000.0;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp   = 3600.0/rho_cur;   // speed of sound squared

//  cout << "rho_cur = " << rho_cur << " press = " << pressure << " dp_drho = " << dp_drho << endl;

}
