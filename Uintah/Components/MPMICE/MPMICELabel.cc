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
  cVelocityLabel = scinew VarLabel( "c.velocity",
			CCVariable<Vector>::getTypeDescription() );
  cMassLabel = scinew VarLabel( "c.mass",
			CCVariable<double>::getTypeDescription() );
} 

MPMICELabel::~MPMICELabel()
{
  delete cVelocityLabel;
  delete cMassLabel;
}

// $Log$
// Revision 1.1  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//
