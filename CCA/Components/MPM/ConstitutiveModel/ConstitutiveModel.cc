
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
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


//for coupling to ICE
double ConstitutiveModel::computeRhoMicro(double pressure)
{

  // This is really only valid now for CompNeoHook(Plas)
  double rho_orig = 1000.0;
  double rho_cur;
  double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double bulk = 2000.0;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

//  cout << rho_cur << endl;

  return rho_cur;
}

void ConstitutiveModel::computePressEOS(const double  rho_cur, double& pressure,                                              double& dp_drho, double& ss_new)
{

  // This is really only valid now for CompNeoHook(Plas)
  double rho_orig = 1000.0;
  double p_ref=101325.0;
  double bulk = 2000.0;

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  ss_new   = sqrt(bulk/rho_cur);

//  cout << "rho_cur = " << rho_cur << " press = " << pressure << " dp_drho = " << dp_drho << endl;

}


