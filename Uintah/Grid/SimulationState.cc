
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>

using namespace Uintah::Grid;

SimulationState::SimulationState()
{
   delt_label = new VarLabel("delt", ReductionVariable<double>::getTypeDescription());
}

//
// $Log$
// Revision 1.1  2000/04/20 18:56:30  sparker
// Updates to MPM
//
//

