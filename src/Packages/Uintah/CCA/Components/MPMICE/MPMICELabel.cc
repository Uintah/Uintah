#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
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
  velstar_CCLabel  = scinew VarLabel( "velstar_CC",
			CCVariable<Vector>::getTypeDescription() );
  dvdt_CCLabel     = scinew VarLabel( "dvdt_CC",
			CCVariable<Vector>::getTypeDescription() );
  dTdt_CCLabel     = scinew VarLabel( "dTdt_CC",
			CCVariable<double>::getTypeDescription() );
  temp_CCLabel     = scinew VarLabel("temp_CC",
			CCVariable<double>::getTypeDescription() );
  temp_CC_scratchLabel = scinew VarLabel("temp_CC_scratch",
			CCVariable<double>::getTypeDescription() );
  press_NCLabel      = scinew VarLabel("pressureNC",
			NCVariable<double>::getTypeDescription());

  velInc_CCLabel     = scinew VarLabel("velIncCC",
			CCVariable<Vector>::getTypeDescription());
  velInc_NCLabel     = scinew VarLabel("velIncNC",
			NCVariable<Vector>::getTypeDescription());

  burnedMassCCLabel   = scinew VarLabel("burnedMass",
			CCVariable<double>::getTypeDescription());  
  releasedHeatCCLabel = scinew VarLabel("releasedHeat",
			CCVariable<double>::getTypeDescription());
  sumBurnedMassLabel  = scinew VarLabel("sumBurnedMass",
			CCVariable<double>::getTypeDescription());  
  sumReleasedHeatLabel = scinew VarLabel("sumReleasedHeat",
			CCVariable<double>::getTypeDescription());
  sumCreatedVolLabel   = scinew VarLabel("sumCreateVol",
                        CCVariable<double>::getTypeDescription());


} 

MPMICELabel::~MPMICELabel()
{
  delete cMassLabel;
  delete cVolumeLabel;
  delete vel_CCLabel;
  delete velstar_CCLabel;
  delete dvdt_CCLabel;
  delete dTdt_CCLabel;
  delete temp_CCLabel;
  delete temp_CC_scratchLabel;
  delete press_NCLabel;
  delete velInc_CCLabel;
  delete velInc_NCLabel;
  delete burnedMassCCLabel;
  delete releasedHeatCCLabel;
  delete sumBurnedMassLabel;
  delete sumReleasedHeatLabel;
  delete sumCreatedVolLabel;
}
