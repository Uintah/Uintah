#include <Uintah/Components/ICE/EOS/IdealGas.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/ICE/ICEMaterial.h>

using namespace Uintah::ICESpace;

IdealGas::IdealGas(ProblemSpecP& ps)
{
   // Constructor
  ps->require("gas_constant",d_gas_constant);
  lb = scinew ICELabel();

}

IdealGas::~IdealGas()
{
  delete lb;
}


double IdealGas::getGasConstant() const
{
  return d_gas_constant;
}

void IdealGas::initializeEOSData(const Patch* patch, const ICEMaterial* matl,
			    DataWarehouseP& new_dw)
{
}

void IdealGas::addComputesAndRequiresSS(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  int dwindex = matl->getDWIndex();
  task->requires(old_dw,lb->temp_CCLabel,      dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->rho_micro_CCLabel, dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,        dwindex, patch,Ghost::None);
  task->computes(new_dw,lb->speedSound_CCLabel,dwindex, patch);

}

void IdealGas::addComputesAndRequiresRM(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  int dwindex = matl->getDWIndex();
  task->requires(old_dw,lb->temp_CCLabel,     dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->press_CCLabel,    dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,       dwindex, patch,Ghost::None);
  task->computes(new_dw,lb->rho_micro_CCLabel,dwindex, patch);

}

void IdealGas::addComputesAndRequiresPEOS(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  int dwindex = matl->getDWIndex();
  task->requires(old_dw,lb->rho_micro_CCLabel, dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->temp_CCLabel,      dwindex, patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,        dwindex, patch,Ghost::None);
  task->computes(new_dw,lb->press_CCLabel,     dwindex, patch);

}



void IdealGas::computeSpeedSound(const Patch* patch,
                                 const ICEMaterial* matl,
                                 DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw)
{
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> speedSound;

  int dwindex = matl->getDWIndex();
  double gamma = matl->getGamma();

  old_dw->get(temp,      lb->temp_CCLabel,      dwindex,patch,Ghost::None, 0); 
  old_dw->get(rho_micro, lb->rho_micro_CCLabel, dwindex,patch,Ghost::None, 0); 
  old_dw->get(cv,        lb->cv_CCLabel,        dwindex,patch,Ghost::None, 0); 
  new_dw->allocate(speedSound,lb->speedSound_CCLabel,dwindex,patch);

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    double dp_drho  = (gamma - 1.0)   * cv[*iter] * temp[*iter];
    double dp_de    = (gamma - 1.0)   * rho_micro[*iter];
    double press    = (gamma - 1.0)   * rho_micro[*iter]*cv[*iter]*temp[*iter];
    double denom    = rho_micro[*iter]*rho_micro[*iter];
    speedSound[*iter] =  sqrt(dp_drho + dp_de* (press/(denom)));
#if 0
    cout << "speedSound"<<*iter<<"="<<speedSound[*iter]<<endl;
#endif
  }

  new_dw->put(speedSound,lb->speedSound_CCLabel,dwindex,patch);
}

double IdealGas::computeRhoMicro(double& press, double& gamma,
				 double& cv, double& Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

void IdealGas::computePressEOS(double& rhoM, double& gamma,
			       double& cv, double& Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}


void IdealGas::computeRhoMicro(const Patch* patch,
			       const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{

  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> press;
  
  int dwindex = matl->getDWIndex();

  old_dw->get(temp,  lb->temp_CCLabel,  dwindex,patch,Ghost::None, 0); 
  old_dw->get(cv,    lb->cv_CCLabel  ,  dwindex,patch,Ghost::None, 0); 
  old_dw->get(press, lb->press_CCLabel, 0,     patch,Ghost::None, 0); 
  new_dw->allocate(rho_micro,lb->rho_micro_CCLabel,dwindex,patch);

  double gamma = matl->getGamma();

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    rho_micro[*iter] = press[*iter]/((gamma -1.)*cv[*iter]*temp[*iter]);
  }

  new_dw->put(rho_micro,lb->rho_micro_CCLabel,dwindex,patch);
}

void IdealGas::computePressEOS(const Patch* patch,
                               const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{
  int dwindex = matl->getDWIndex();
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> press;

  double gamma = matl->getGamma();

  old_dw->get(temp,      lb->temp_CCLabel,      dwindex,patch,Ghost::None, 0); 
  old_dw->get(cv,        lb->cv_CCLabel,        dwindex,patch,Ghost::None, 0); 
  old_dw->get(rho_micro, lb->rho_micro_CCLabel, dwindex,patch,Ghost::None, 0); 
  new_dw->allocate(press,lb->press_CCLabel,     dwindex,patch);


  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    press[*iter] = (gamma - 1.)* rho_micro[*iter] * cv[*iter] * temp[*iter];
  }

  new_dw->put(press,lb->press_CCLabel,dwindex,patch);
}


//$Log$
//Revision 1.11  2000/12/18 23:25:56  jas
//2d ice works for simple advection.
//
//Revision 1.10  2000/11/28 03:50:30  jas
//Added {X,Y,Z}FCVariables.  Things still don't work yet!
//
//Revision 1.9  2000/11/15 00:51:55  guilkey
//Changed code to take advantage of the ICEMaterial stuff I committed
//recently in preparation for coupling the two codes.
//
//Revision 1.8  2000/11/14 04:02:12  jas
//Added getExtraCellIterator and things now appear to be working up to
//face centered velocity calculations.
//
//Revision 1.7  2000/10/31 04:14:28  jas
//Added stiff gas EOS type.  It is just a copy of IdealGas.
//
//Revision 1.6  2000/10/27 23:39:54  jas
//Added gas constant to lookup.
//
//Revision 1.5  2000/10/14 02:49:50  jas
//Added implementation of compute equilibration pressure.  Still need to do
//the update of BCS and hydrostatic pressure.  Still some issues with
//computes and requires - will compile but won't run.
//
//Revision 1.4  2000/10/10 22:18:27  guilkey
//Added some simple functions
//
//Revision 1.3  2000/10/10 20:35:12  jas
//Move some stuff around.
//
//Revision 1.2  2000/10/09 22:37:04  jas
//Cleaned up labels and added more computes and requires for EOS.
//
//Revision 1.1  2000/10/06 04:02:16  jas
//Move into a separate EOS directory.
//
//Revision 1.3  2000/10/06 03:47:26  jas
//Added computes for the initialization so that step 1 works.  Added a couple
//of CC labels for step 1. Can now go thru multiple timesteps doing work
//only in step 1.
//
//Revision 1.2  2000/10/05 04:26:48  guilkey
//Added code for part of the EOS evaluation.
//
//Revision 1.1  2000/10/04 23:40:12  jas
//The skeleton framework for an EOS model.  Does nothing.
//
