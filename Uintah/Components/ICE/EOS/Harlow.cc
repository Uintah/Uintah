#include <Uintah/Components/ICE/EOS/Harlow.h>
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

Harlow::Harlow(ProblemSpecP& ps)
{
   // Constructor
  lb = scinew ICELabel();

}

Harlow::~Harlow()
{
  delete lb;
}

void Harlow::initializeEOSData(const Patch* patch, const ICEMaterial* matl,
			    DataWarehouseP& new_dw)
{
}

void Harlow::addComputesAndRequiresSS(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  task->requires(old_dw,lb->temp_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->rho_micro_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->computes(new_dw,lb->speedSound_CCLabel,matl->getDWIndex(), patch);

}

void Harlow::addComputesAndRequiresRM(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  task->requires(old_dw,lb->temp_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->press_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->computes(new_dw,lb->rho_micro_CCLabel,matl->getDWIndex(), patch);

}

void Harlow::addComputesAndRequiresPEOS(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{
  task->requires(old_dw,lb->rho_micro_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->temp_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->computes(new_dw,lb->press_CCLabel,matl->getDWIndex(), patch);

}



void Harlow::computeSpeedSound(const Patch* patch,
                                 const ICEMaterial* matl,
                                 DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw)
{
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> speedSound;

  int vfindex = matl->getVFIndex();
  double gamma = matl->getGamma();

  old_dw->get(temp, lb->temp_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(rho_micro, lb->rho_micro_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(cv, lb->cv_CCLabel, vfindex,patch,Ghost::None, 0); 
  new_dw->allocate(speedSound,lb->speedSound_CCLabel,vfindex,patch);


  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    double dp_drho = (gamma - 1.0) * cv[*iter] * temp[*iter];
    double dp_de   = (gamma - 1.0) * rho_micro[*iter];
    double press   = (gamma - 1.0) * rho_micro[*iter]*cv[*iter]*temp[*iter];
    double denom = rho_micro[*iter]*rho_micro[*iter];
    speedSound[*iter] =  sqrt(dp_drho + dp_de* (press/(denom*denom)));
  }

  new_dw->put(speedSound,lb->speedSound_CCLabel,vfindex,patch);

}

double Harlow::computeRhoMicro(double& press, double& gamma,
				 double& cv, double& Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

void Harlow::computePressEOS(double& rhoM, double& gamma,
			       double& cv, double& Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}


void Harlow::computeRhoMicro(const Patch* patch,
			       const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{

  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> press;
  
  int vfindex = matl->getVFIndex();

  old_dw->get(temp, lb->temp_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(cv, lb->cv_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(press, lb->press_CCLabel, vfindex,patch,Ghost::None, 0); 
  new_dw->allocate(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);

  double gamma = matl->getGamma();

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    rho_micro[*iter] = press[*iter]/((gamma -1.)*cv[*iter]*temp[*iter]);
  }

  new_dw->put(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);


}

void Harlow::computePressEOS(const Patch* patch,
                               const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{

  int vfindex = matl->getVFIndex();
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> press;

  double gamma = matl->getGamma();

  old_dw->get(temp, lb->temp_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(cv, lb->cv_CCLabel, vfindex,patch,Ghost::None, 0); 
  old_dw->get(rho_micro, lb->cv_CCLabel, vfindex,patch,Ghost::None, 0); 
  new_dw->allocate(press,lb->rho_micro_CCLabel,vfindex,patch);


  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    press[*iter] = (gamma - 1.)* rho_micro[*iter] * cv[*iter] * temp[*iter];
  }

  new_dw->put(press,lb->press_CCLabel,vfindex,patch);

}


//$Log$
//Revision 1.2  2000/11/14 04:02:12  jas
//Added getExtraCellIterator and things now appear to be working up to
//face centered velocity calculations.
//
//Revision 1.1  2000/10/26 23:37:05  jas
//Added Harlow EOS.
//

