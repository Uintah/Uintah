#include <Uintah/Components/ICE/EOS/EquationOfState.h>
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
//Revision 1.1.2.1  2000/10/19 05:17:41  sparker
//Merge changes from main branch into csafe_risky1
//
//Revision 1.1  2000/10/06 04:02:16  jas
//Move into a separate EOS directory.
//
//Revision 1.2  2000/10/04 20:17:52  jas
//Change namespace ICE to ICESpace.
//
//Revision 1.1  2000/10/04 19:26:14  guilkey
//Initial commit of some classes to help mainline ICE.
//
