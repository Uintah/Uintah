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
}

//
// $Log$
// Revision 1.1  2000/06/20 17:59:12  tan
// Heat Conduction model created to move heat conduction part of code from MPM.
// Thus make MPM clean and easy to maintain.
//
