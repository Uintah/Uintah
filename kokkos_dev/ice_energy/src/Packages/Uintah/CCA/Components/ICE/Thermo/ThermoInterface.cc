
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

using namespace Uintah;

ThermoInterface::ThermoInterface(ICEMaterial* ice_matl)
  : PropertyBase(ice_matl)
{
}

ThermoInterface::~ThermoInterface()
{
}
