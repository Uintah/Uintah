#include <Uintah/Components/ICE/IdealGas.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>

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
			    DataWarehouseP& ndw_dw)
{
}

void IdealGas::addComputesAndRequires(Task* task,
				 const ICEMaterial* matl, const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) const
{

}




//$Log$
//Revision 1.1  2000/10/04 23:40:12  jas
//The skeleton framework for an EOS model.  Does nothing.
//
