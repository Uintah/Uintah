#include <Uintah/Components/MPMICE/MPMICELabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace Uintah::MPMICESpace;

MPMICELabel::MPMICELabel()
{
  // Cell centered variables
  cMassLabel    = scinew VarLabel( "c.mass",
			CCVariable<double>::getTypeDescription() );
  cVolumeLabel  = scinew VarLabel( "c.volume",
			CCVariable<double>::getTypeDescription() );
  vel_CCLabel   = scinew VarLabel( "vel_CC",
			CCVariable<Vector>::getTypeDescription() );
  mom_L_CCLabel = scinew VarLabel( "mom_L_CC",
			CCVariable<Vector>::getTypeDescription() );
  dvdt_CCLabel  = scinew VarLabel( "dvdt_CC",
			CCVariable<Vector>::getTypeDescription() );
} 

MPMICELabel::~MPMICELabel()
{
  delete cMassLabel;
  delete cVolumeLabel;
  delete vel_CCLabel;
  delete mom_L_CCLabel;
  delete dvdt_CCLabel;
}

// $Log$
// Revision 1.4  2001/01/15 23:21:54  guilkey
// Cleaned up CCMomentum exchange, so it now looks more like Todd's.
// Added effects back to solid material.  Need NodeIterator to be fixed,
// and need to figure out how to apply BCs from the ICE code.
//
// Revision 1.3  2001/01/14 02:30:01  guilkey
// CC momentum exchange now works from solid to fluid, still need to
// add fluid to solid effects.
//
// Revision 1.2  2001/01/11 20:11:16  guilkey
// Working on getting momentum exchange to work.  It doesnt' yet.
//
// Revision 1.1  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//
