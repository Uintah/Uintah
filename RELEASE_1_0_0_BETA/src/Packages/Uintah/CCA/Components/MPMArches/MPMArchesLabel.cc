#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
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

MPMArchesLabel::MPMArchesLabel()
{
  // Cell centered variables
  cMassLabel    = scinew VarLabel( "c.mass",
			CCVariable<double>::getTypeDescription() );
  cVolumeLabel  = scinew VarLabel( "c.volume",
			CCVariable<double>::getTypeDescription() );
  vel_CCLabel   = scinew VarLabel( "vel_CC",
			CCVariable<Vector>::getTypeDescription() );
  void_frac_CCLabel  = scinew VarLabel( "void_frac_CC",
			CCVariable<double>::getTypeDescription() );

  // drag momentum acting on x-face
  momExDragForceFCXLabel = scinew VarLabel("momExDragForceFCX",
				 SFCXVariable<Vector>::getTypeDescription() );
  // pressure force acting on x-face
  momExPressureForceFCXLabel = scinew VarLabel("momExPressureForceFCX",
				 SFCXVariable<Vector>::getTypeDescription() );
  // drag momentum acting on y-face
  momExDragForceFCYLabel = scinew VarLabel("momExDragForceFCY",
				 SFCYVariable<Vector>::getTypeDescription() );
  // pressure force acting on y-face
  momExPressureForceFCYLabel = scinew VarLabel("momExPressureForceFCY",
				 SFCYVariable<Vector>::getTypeDescription() );
  // drag momentum acting on z-face
  momExDragForceFCZLabel = scinew VarLabel("momExDragForceFCZ",
				 SFCZVariable<Vector>::getTypeDescription() );
  // pressure force acting on z-face
  momExPressureForceFCZLabel = scinew VarLabel("momExPressureForceFCZ",
				 SFCZVariable<Vector>::getTypeDescription() );

  // U-Velocity Linear Src Labels from multi-material
  d_uVel_mmLinSrcLabel = scinew VarLabel("uVel_mmLinSrc",
				      SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels from multi-material
  d_uVel_mmNonlinSrcLabel = scinew VarLabel("uVel_mmNonlinSrc",
				    SFCXVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels from multi-material
  d_vVel_mmLinSrcLabel = scinew VarLabel("vVel_mmLinSrc",
				      SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels from multi-material
  d_vVel_mmNonlinSrcLabel = scinew VarLabel("vVel_mmNonlinSrc",
				    SFCYVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels from multi-material
  d_wVel_mmLinSrcLabel = scinew VarLabel("wVel_mmLinSrc",
				      SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels from multi-material
  d_wVel_mmNonlinSrcLabel = scinew VarLabel("wVel_mmNonlinSrc",
				    SFCZVariable<double>::getTypeDescription() );

   // produced by scheduleInterpolateCCToFC
   // if these are duplicates of some of the above, feel free to
   // straighten this out
   d_xMomFCLabel = scinew VarLabel("xMomFC",
				SFCXVariable<double>::getTypeDescription() );

   d_yMomFCLabel = scinew VarLabel("yMomFC",
                                SFCYVariable<double>::getTypeDescription() );

   d_zMomFCLabel = scinew VarLabel("zMomFC",
                                SFCZVariable<double>::getTypeDescription() );

} 

MPMArchesLabel::~MPMArchesLabel()
{
  delete void_frac_CCLabel;
  delete cMassLabel;
  delete cVolumeLabel;
  delete vel_CCLabel;
  delete momExDragForceFCXLabel;
  delete momExPressureForceFCXLabel;
  delete momExDragForceFCYLabel;
  delete momExPressureForceFCYLabel;
  delete momExDragForceFCZLabel;
  delete momExPressureForceFCZLabel;
  delete d_uVel_mmLinSrcLabel;
  delete d_uVel_mmNonlinSrcLabel;
  delete d_vVel_mmLinSrcLabel;
  delete d_vVel_mmNonlinSrcLabel;
  delete d_wVel_mmLinSrcLabel;
  delete d_wVel_mmNonlinSrcLabel;

  delete d_xMomFCLabel;
  delete d_yMomFCLabel;
  delete d_zMomFCLabel;
}
