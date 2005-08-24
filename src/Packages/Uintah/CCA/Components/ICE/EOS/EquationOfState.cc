#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>

using namespace Uintah;

EquationOfState::EquationOfState(ICEMaterial* ice_matl)
  : PropertyBase(ice_matl)
{
   // Constructor
}

EquationOfState::~EquationOfState()
{
}
