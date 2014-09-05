#include <Packages/Uintah/CCA/Components/ICE/EOS/StiffGas.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>

using namespace Uintah;

StiffGas::StiffGas(ProblemSpecP& ps)
{
   // Constructor
  lb = scinew ICELabel();

}

StiffGas::~StiffGas()
{
  delete lb;
}

void StiffGas::initializeEOSData(const Patch* patch, const ICEMaterial* matl,
			    DataWarehouseP& new_dw)
{
}

void StiffGas::addComputesAndRequiresSS(Task* task,
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

void StiffGas::addComputesAndRequiresRM(Task* task,
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

void StiffGas::addComputesAndRequiresPEOS(Task* task,
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



void StiffGas::computeSpeedSound(const Patch* patch,
                                 const ICEMaterial* matl,
                                 DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw)
{
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> speedSound;

  int vfindex = matl->getDWIndex();
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

double StiffGas::computeRhoMicro(double& press, double& gamma,
				 double& cv, double& Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}
//__________________________________
//
void StiffGas::computeTempCC(const Patch* patch,
			     const CCVariable<double>& press, 
			     const double& gamma,
			     const double& cv,
			     const CCVariable<double>& rho_micro, 
			     CCVariable<double>& Temp)
{
  for (CellIterator iter = patch->getCellIterator();!iter.done();iter++) { 
    Temp[*iter]= press[*iter]/ ( (gamma - 1.0) * cv * rho_micro[*iter] );
  }
}

void StiffGas::computePressEOS(double& rhoM, double& gamma,
			       double& cv, double& Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}


void StiffGas::computeRhoMicro(const Patch* patch,
			       const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{

  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> press;

  int vfindex = matl->getDWIndex();

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

void StiffGas::computePressEOS(const Patch* patch,
                               const ICEMaterial* matl,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw)
{

  int vfindex = matl->getDWIndex();
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
