#include <Packages/Uintah/CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <vector>

using namespace Uintah;

HeatConduction::HeatConduction(ProblemSpecP&,SimulationStateP& d_sS)
{
  d_sharedState = d_sS;
}

