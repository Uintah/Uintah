#include <Uintah/Components/ICE/IdealGas.h>
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
  lb = scinew ICELabel();

}

IdealGas::~IdealGas()
{
  delete lb;
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
  task->requires(old_dw,lb->temp_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->rho_micro_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->requires(old_dw,lb->cv_CCLabel,
		matl->getDWIndex(),patch,Ghost::None);
  task->computes(new_dw,lb->speedSound_CCLabel,matl->getDWIndex(), patch);

}

void IdealGas::addComputesAndRequiresCEB(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{

}

void IdealGas::computeSpeedSound(const Patch* patch,
                                 const ICEMaterial* matl,
                                 DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw)
{
  int vfindex = matl->getVFIndex();
  CCVariable<double> rho_micro;
  CCVariable<double> temp;
  CCVariable<double> cv;
  CCVariable<double> speedSound;

  double gamma = matl->getGamma();

  old_dw->get(temp, lb->temp_CCLabel, vfindex,patch,
						Ghost::None, 0); 
  old_dw->get(rho_micro, lb->rho_micro_CCLabel, vfindex,patch,
						Ghost::None, 0); 
  old_dw->get(cv, lb->cv_CCLabel, vfindex,patch,
						Ghost::None, 0); 
  new_dw->allocate(speedSound,lb->speedSound_CCLabel,vfindex,patch);

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    double dp_drho = (gamma - 1.0) * cv[*iter] * temp[*iter];
    double dp_de   = (gamma - 1.0) * rho_micro[*iter];
    double press   = (gamma - 1.0) * rho_micro[*iter]*cv[*iter]*temp[*iter];
    double denom = rho_micro[*iter]*rho_micro[*iter];
    speedSound[*iter] =  sqrt(dp_drho + dp_de* (press/(denom*denom)));
  }

  new_dw->put(speedSound,lb->speedSound_CCLabel,vfindex,patch);

}

void IdealGas::computeEquilibrationPressure(const Patch* patch,
                                            const ICEMaterial* matl,
                                            DataWarehouseP& old_dw,
                                            DataWarehouseP& new_dw)
{
  int vfindex = matl->getVFIndex();

}

//$Log$
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
