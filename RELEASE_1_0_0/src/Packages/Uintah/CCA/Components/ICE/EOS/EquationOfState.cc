#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>

using namespace Uintah;

EquationOfState::EquationOfState()
{
   // Constructor
  lb = scinew ICELabel();

}

EquationOfState::~EquationOfState()
{
  delete lb;
}
