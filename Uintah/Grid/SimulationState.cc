
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>

using namespace Uintah::Grid;

SimulationState::SimulationState()
{
   delt_label = new VarLabel("delt", ReductionVariable<double>::getTypeDescription());
}

void SimulationState::registerMaterial(Material* matl)
{
   matls.push_back(matl);
}

//
// $Log$
// Revision 1.2  2000/04/24 21:04:38  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.1  2000/04/20 18:56:30  sparker
// Updates to MPM
//
//

