#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

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
