#include <Uintah/Components/MPM/ThermalContact/ThermalContact.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::MPM;

ThermalContact::ThermalContact()
{
}

void ThermalContact::computeHeatExchange(const ProcessorContext*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
}

void ThermalContact::initializeThermalContact(const Patch* patch,
					int vfindex,
					DataWarehouseP& new_dw)
{
}

void ThermalContact::addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
}


//
// $Log$
// Revision 1.1  2000/05/31 18:17:27  tan
// Create ThermalContact class to handle heat exchange in
// contact mechanics.
//
//


