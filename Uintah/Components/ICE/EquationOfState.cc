#include <Uintah/Components/ICE/EquationOfState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::ICESpace;

EquationOfState::EquationOfState()
{
   // Constructor
  lb = scinew ICELabel();

}

EquationOfState::~EquationOfState()
{
  delete lb;
}

//$Log$
//Revision 1.2  2000/10/04 20:17:52  jas
//Change namespace ICE to ICESpace.
//
//Revision 1.1  2000/10/04 19:26:14  guilkey
//Initial commit of some classes to help mainline ICE.
//
