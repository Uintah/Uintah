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
  cMomentumLabel = scinew VarLabel( "c.momentum",
			CCVariable<Vector>::getTypeDescription() );
  cMassLabel = scinew VarLabel( "c.mass",
			CCVariable<double>::getTypeDescription() );
} 

MPMICELabel::~MPMICELabel()
{
  delete cMomentumLabel;
  delete cMassLabel;
}

// $Log$
// Revision 1.2  2001/01/11 20:11:16  guilkey
// Working on getting momentum exchange to work.  It doesnt' yet.
//
// Revision 1.1  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//
