#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
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
  mom_L_CCLabel    = scinew VarLabel( "mom_L_CC",
			CCVariable<Vector>::getTypeDescription() );
  dvdt_CCLabel     = scinew VarLabel( "dvdt_CC",
			CCVariable<Vector>::getTypeDescription() );
  temp_CCLabel     = scinew VarLabel("temp_CC",
			CCVariable<double>::getTypeDescription() );
  vol_frac_CCLabel = scinew VarLabel("vol_frac_CC",
			CCVariable<double>::getTypeDescription());
  cv_CCLabel       = scinew VarLabel("cv_CC",
			CCVariable<double>::getTypeDescription());
  rho_CCLabel      = scinew VarLabel("rho_CC",
			CCVariable<double>::getTypeDescription());
  rho_micro_CCLabel  = scinew VarLabel("rho_micro_CC",
			CCVariable<double>::getTypeDescription());
  speedSound_CCLabel = scinew VarLabel("speedSound_CC",
			CCVariable<double>::getTypeDescription());
  mom_source_CCLabel = scinew VarLabel("mom_source_CC",
			CCVariable<Vector>::getTypeDescription());

    uvel_FCLabel       =
     scinew VarLabel("uvel_FC",   SFCXVariable<double>::getTypeDescription() );
    vvel_FCLabel       =
     scinew VarLabel("vvel_FC",   SFCYVariable<double>::getTypeDescription() );
    wvel_FCLabel       =
     scinew VarLabel("wvel_FC",   SFCZVariable<double>::getTypeDescription() );
    uvel_FCMELabel       =
     scinew VarLabel("uvel_FCME", SFCXVariable<double>::getTypeDescription() );
    vvel_FCMELabel       =
     scinew VarLabel("vvel_FCME", SFCYVariable<double>::getTypeDescription() );
    wvel_FCMELabel       =
     scinew VarLabel("wvel_FCME", SFCZVariable<double>::getTypeDescription() );

} 

MPMICELabel::~MPMICELabel()
{
  delete cMassLabel;
  delete cVolumeLabel;
  delete vel_CCLabel;
  delete velstar_CCLabel;
  delete mom_L_CCLabel;
  delete dvdt_CCLabel;
  delete temp_CCLabel;
  delete speedSound_CCLabel;
  delete cv_CCLabel;
  delete rho_CCLabel;
  delete rho_micro_CCLabel;
  delete vol_frac_CCLabel;
  delete mom_source_CCLabel;

  delete uvel_FCLabel;
  delete vvel_FCLabel;
  delete wvel_FCLabel;
  delete uvel_FCMELabel;
  delete vvel_FCMELabel;
  delete wvel_FCMELabel;
}
