#include <Uintah/Components/MPM/HeatConduction/HeatConduction.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/Task.h>

#include <vector>

using namespace Uintah::MPM;

HeatConduction::HeatConduction(ProblemSpecP& ps,SimulationStateP& d_sS)
{
  d_sharedState = d_sS;

  ps->require("thermal_conductivity",d_thermalConductivity);
  ps->require("specific_heat",d_specificHeat);
  ps->require("heat_transfer_coefficient",d_heatTransferCoefficient);
}

double HeatConduction::getThermalConductivity() const
{
  return d_thermalConductivity;
}

double HeatConduction::getSpecificHeat() const
{
  return d_specificHeat;
}

double HeatConduction::getHeatTransferCoefficient() const
{
  return d_heatTransferCoefficient;
}


//
// $Log$
// Revision 1.2  2000/06/22 22:36:17  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.1  2000/06/20 17:59:12  tan
// Heat Conduction model created to move heat conduction part of code from MPM.
// Thus make MPM clean and easy to maintain.
//
