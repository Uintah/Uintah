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
  // Cell centered variables for solid materials

  cMassLabel             = VarLabel::create( "c.mass",
					    CCVariable<double>::getTypeDescription() );
  cVolumeLabel           = VarLabel::create( "c.volume",
					    CCVariable<double>::getTypeDescription() );
  solid_fraction_CCLabel = VarLabel::create( "solid_fraction",
					    CCVariable<double>::getTypeDescription() );

  // Cell-centered solid velocity labels

  vel_CCLabel            = VarLabel::create( "vel_CC",
					    CCVariable<Vector>::getTypeDescription() );
  xvel_CCLabel = VarLabel::create("xvel_CC", 
				 CCVariable<double>::getTypeDescription() );
  yvel_CCLabel = VarLabel::create("yvel_CC", 
				 CCVariable<double>::getTypeDescription() );
  zvel_CCLabel = VarLabel::create("zvel_CC", 
				 CCVariable<double>::getTypeDescription() );

  // face-centered solid velocity labels

  xvel_FCXLabel = VarLabel::create("xvel_FCX",
			     SFCXVariable<double>::getTypeDescription() );
  xvel_FCYLabel = VarLabel::create("xvel_FCY",
			     SFCYVariable<double>::getTypeDescription() );
  xvel_FCZLabel = VarLabel::create("xvel_FCZ",
			     SFCZVariable<double>::getTypeDescription() );

  yvel_FCXLabel = VarLabel::create("yvel_FCX",
			     SFCXVariable<double>::getTypeDescription() );
  yvel_FCYLabel = VarLabel::create("yvel_FCY",
			     SFCYVariable<double>::getTypeDescription() );
  yvel_FCZLabel = VarLabel::create("yvel_FCZ",
			     SFCZVariable<double>::getTypeDescription() );

  zvel_FCXLabel = VarLabel::create("zvel_FCX",
			     SFCXVariable<double>::getTypeDescription() );
  zvel_FCYLabel = VarLabel::create("zvel_FCY",
			     SFCYVariable<double>::getTypeDescription() );
  zvel_FCZLabel = VarLabel::create("zvel_FCZ",
			     SFCZVariable<double>::getTypeDescription() );

  // labels for solid material calculations:

  SumAllForcesCCLabel = VarLabel::create("SumAllForcesCC",
				 CCVariable<Vector>::getTypeDescription() );

  AccArchesCCLabel = VarLabel::create("AccArchesCC",
				 CCVariable<Vector>::getTypeDescription() );

  SumAllForcesNCLabel = VarLabel::create("SumAllForcesNC",
				 NCVariable<Vector>::getTypeDescription() );

  AccArchesNCLabel = VarLabel::create("AccArchesNC",
				 NCVariable<Vector>::getTypeDescription() );

  // used for temporary computation:

  // CC

  // x-drag momentum acting at cell center
  DragForceX_CCLabel = VarLabel::create("DragForceX_CC",
				 CCVariable<double>::getTypeDescription() );
  // y-drag momentum acting at cell center
  DragForceY_CCLabel = VarLabel::create("DragForceY_CC",
				 CCVariable<double>::getTypeDescription() );
  // z-drag momentum acting at cell center
  DragForceZ_CCLabel = VarLabel::create("DragForceZ_CC",
				 CCVariable<double>::getTypeDescription() );

  // Passed to MPM:

  // Drag Forces:

  // FCX

#if 0
  // x-drag momentum acting on x-face
  DragForceX_FCXLabel = VarLabel::create("DragForceX_FCX",
				 SFCXVariable<double>::getTypeDescription() );
#endif
  // y-drag momentum acting on x-face
  DragForceY_FCXLabel = VarLabel::create("DragForceY_FCX",
				 SFCXVariable<double>::getTypeDescription() );
  // z-drag momentum acting on x-face
  DragForceZ_FCXLabel = VarLabel::create("DragForceZ_FCX",
				 SFCXVariable<double>::getTypeDescription() );

  // FCY

  // x-drag momentum acting on y-face
  DragForceX_FCYLabel = VarLabel::create("DragForceX_FCY",
				 SFCYVariable<double>::getTypeDescription() );
#if 0
  // y-drag momentum acting on y-face
  DragForceY_FCYLabel = VarLabel::create("DragForceY_FCY",
				 SFCYVariable<double>::getTypeDescription() );
#endif
  // z-drag momentum acting on y-face
  DragForceZ_FCYLabel = VarLabel::create("DragForceZ_FCY",
				 SFCYVariable<double>::getTypeDescription() );

  // FCZ

  // x-drag momentum acting on z-face
  DragForceX_FCZLabel = VarLabel::create("DragForceX_FCZ",
				 SFCZVariable<double>::getTypeDescription() );
  // y-drag momentum acting on z-face
  DragForceY_FCZLabel = VarLabel::create("DragForceY_FCZ",
				 SFCZVariable<double>::getTypeDescription() );
#if 0
  // z-drag momentum acting on z-face
  DragForceZ_FCZLabel = VarLabel::create("DragForceZ_FCZ",
				 SFCZVariable<double>::getTypeDescription() );
#endif

  // Pressure Forces:

  // pressure force acting on x-face
  PressureForce_FCXLabel = VarLabel::create("PressureForce_FCX",
				 SFCXVariable<double>::getTypeDescription() );
  // pressure force acting on y-face
  PressureForce_FCYLabel = VarLabel::create("PressureForce_FCY",
				 SFCYVariable<double>::getTypeDescription() );
  // pressure force acting on z-face
  PressureForce_FCZLabel = VarLabel::create("PressureForce_FCZ",
				 SFCZVariable<double>::getTypeDescription() );

  // labels for gas calculations

  // Gas void fraction
  void_frac_CCLabel  = VarLabel::create( "void_frac_CC",
			CCVariable<double>::getTypeDescription() );

  // u-velocity source term labels


  // U-Velocity Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_uVel_mmLinSrc_CCLabel = VarLabel::create("uVel_mmLinSrc_CC",
				      CCVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_uVel_mmLinSrc_CC_CollectLabel = VarLabel::create("uVel_mmLinSrc_CC_Collect",
				      CCVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels from multi-material (FCX)
  // used in actual summation of source
  d_uVel_mmLinSrcLabel = VarLabel::create("uVel_mmLinSrc",
				      SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels from multi-material (FCY)
  // used for temporary computation
  d_uVel_mmLinSrc_FCYLabel = VarLabel::create("uVel_mmLinSrc_FCY",
				      SFCYVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels from multi-material (FCZ)
  // used for temporary computation
  d_uVel_mmLinSrc_FCZLabel = VarLabel::create("uVel_mmLinSrc_FCZ",
				      SFCZVariable<double>::getTypeDescription() );


  // U-Velocity Non Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_uVel_mmNonlinSrc_CCLabel = VarLabel::create("uVel_mmNonlinSrc_CC",
				    CCVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_uVel_mmNonlinSrc_CC_CollectLabel = VarLabel::create("uVel_mmNonlinSrc_CC_Collect",
				    CCVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels from multi-material (FCX)
  // used in actual summation of source
  d_uVel_mmNonlinSrcLabel = VarLabel::create("uVel_mmNonlinSrc",
				    SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels from multi-material (FCY)
  // used for temporary computation
  d_uVel_mmNonlinSrc_FCYLabel = VarLabel::create("uVel_mmNonlinSrc_FCY",
				    SFCYVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels from multi-material (FCZ)
  // used for temporary computation
  d_uVel_mmNonlinSrc_FCZLabel = VarLabel::create("uVel_mmNonlinSrc_FCZ",
				    SFCZVariable<double>::getTypeDescription() );

  // v-velocity source term labels


  // V-Velocity Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_vVel_mmLinSrc_CCLabel = VarLabel::create("vVel_mmLinSrc_CC",
				      CCVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_vVel_mmLinSrc_CC_CollectLabel = VarLabel::create("vVel_mmLinSrc_CC_Collect",
				      CCVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels from multi-material (FCY)
  // used in actual summation of source
  d_vVel_mmLinSrcLabel = VarLabel::create("vVel_mmLinSrc",
				      SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels from multi-material (FCZ)
  // used for temporary computation
  d_vVel_mmLinSrc_FCZLabel = VarLabel::create("vVel_mmLinSrc_FCZ",
				      SFCZVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels from multi-material (FCX)
  // used for temporary computation
  d_vVel_mmLinSrc_FCXLabel = VarLabel::create("vVel_mmLinSrc_FCX",
				      SFCXVariable<double>::getTypeDescription() );


  // V-Velocity Non Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_vVel_mmNonlinSrc_CCLabel = VarLabel::create("vVel_mmNonlinSrc_CC",
				    CCVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_vVel_mmNonlinSrc_CC_CollectLabel = VarLabel::create("vVel_mmNonlinSrc_CC_Collect",
				    CCVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels from multi-material (FCY)
  // used in actual summation of source
  d_vVel_mmNonlinSrcLabel = VarLabel::create("vVel_mmNonlinSrc",
				    SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels from multi-material (FCZ)
  // used for temporary computation
  d_vVel_mmNonlinSrc_FCZLabel = VarLabel::create("vVel_mmNonlinSrc_FCZ",
				    SFCZVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels from multi-material (FCX)
  // used for temporary computation
  d_vVel_mmNonlinSrc_FCXLabel = VarLabel::create("vVel_mmNonlinSrc_FCX",
				    SFCXVariable<double>::getTypeDescription() );

  // w-velocity source term labels


  // W-Velocity Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_wVel_mmLinSrc_CCLabel = VarLabel::create("wVel_mmLinSrc_CC",
				      CCVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_wVel_mmLinSrc_CC_CollectLabel = VarLabel::create("wVel_mmLinSrc_CC_Collect",
				      CCVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels from multi-material (FCZ)
  // used in actual summation of source
  d_wVel_mmLinSrcLabel = VarLabel::create("wVel_mmLinSrc",
				      SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels from multi-material (FCX)
  // used for temporary computation
  d_wVel_mmLinSrc_FCXLabel = VarLabel::create("wVel_mmLinSrc_FCX",
				      SFCXVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels from multi-material (FCY)
  // used for temporary computation
  d_wVel_mmLinSrc_FCYLabel = VarLabel::create("wVel_mmLinSrc_FCY",
				      SFCYVariable<double>::getTypeDescription() );


  // W-Velocity Non Linear Src Labels from multi-material (CC)
  // used for temporary computation
  d_wVel_mmNonlinSrc_CCLabel = VarLabel::create("wVel_mmNonlinSrc_CC",
				    CCVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels from multi-material (CC) (Collected)
  // used for temporary computation
  d_wVel_mmNonlinSrc_CC_CollectLabel = VarLabel::create("wVel_mmNonlinSrc_CC_Collect",
				    CCVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels from multi-material (FCZ)
  // used in actual summation of source
  d_wVel_mmNonlinSrcLabel = VarLabel::create("wVel_mmNonlinSrc",
				    SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels from multi-material (FCX)
  // used for temporary computation
  d_wVel_mmNonlinSrc_FCXLabel = VarLabel::create("wVel_mmNonlinSrc_FCX",
				    SFCXVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels from multi-material (FCY)
  // used for temporary computation
  d_wVel_mmNonlinSrc_FCYLabel = VarLabel::create("wVel_mmNonlinSrc_FCY",
				    SFCYVariable<double>::getTypeDescription() );

} 

MPMArchesLabel::~MPMArchesLabel()
{
  VarLabel::destroy(cMassLabel);
  VarLabel::destroy(cVolumeLabel);
  VarLabel::destroy(solid_fraction_CCLabel);

  VarLabel::destroy(vel_CCLabel);
  VarLabel::destroy(xvel_CCLabel);
  VarLabel::destroy(yvel_CCLabel);
  VarLabel::destroy(zvel_CCLabel);

  VarLabel::destroy(xvel_FCXLabel);
  VarLabel::destroy(xvel_FCYLabel);
  VarLabel::destroy(xvel_FCZLabel);

  VarLabel::destroy(yvel_FCXLabel);
  VarLabel::destroy(yvel_FCYLabel);
  VarLabel::destroy(yvel_FCZLabel);

  VarLabel::destroy(zvel_FCXLabel);
  VarLabel::destroy(zvel_FCYLabel);
  VarLabel::destroy(zvel_FCZLabel);

  VarLabel::destroy(DragForceX_CCLabel);
  VarLabel::destroy(DragForceY_CCLabel);
  VarLabel::destroy(DragForceZ_CCLabel);

#if 0
  VarLabel::destroy(DragForceX_FCXLabel);
#endif
  VarLabel::destroy(DragForceY_FCXLabel);
  VarLabel::destroy(DragForceZ_FCXLabel);

  VarLabel::destroy(DragForceX_FCYLabel);
#if 0
  VarLabel::destroy(DragForceY_FCYLabel);
#endif
  VarLabel::destroy(DragForceZ_FCYLabel);

  VarLabel::destroy(DragForceX_FCZLabel);
  VarLabel::destroy(DragForceY_FCZLabel);
#if 0
  VarLabel::destroy(DragForceZ_FCZLabel);
#endif

  VarLabel::destroy(PressureForce_FCXLabel);
  VarLabel::destroy(PressureForce_FCYLabel);
  VarLabel::destroy(PressureForce_FCZLabel);

  VarLabel::destroy(SumAllForcesCCLabel);
  VarLabel::destroy(SumAllForcesNCLabel);
  VarLabel::destroy(AccArchesCCLabel);
  VarLabel::destroy(AccArchesNCLabel);

  VarLabel::destroy(void_frac_CCLabel);

  VarLabel::destroy(d_uVel_mmLinSrc_CCLabel);
  VarLabel::destroy(d_uVel_mmLinSrc_CC_CollectLabel);
  VarLabel::destroy(d_uVel_mmLinSrcLabel);
  VarLabel::destroy(d_uVel_mmLinSrc_FCYLabel);
  VarLabel::destroy(d_uVel_mmLinSrc_FCZLabel);
  VarLabel::destroy(d_uVel_mmNonlinSrc_CCLabel);
  VarLabel::destroy(d_uVel_mmNonlinSrc_CC_CollectLabel);
  VarLabel::destroy(d_uVel_mmNonlinSrcLabel);
  VarLabel::destroy(d_uVel_mmNonlinSrc_FCYLabel);
  VarLabel::destroy(d_uVel_mmNonlinSrc_FCZLabel);

  VarLabel::destroy(d_vVel_mmLinSrc_CCLabel);
  VarLabel::destroy(d_vVel_mmLinSrc_CC_CollectLabel);
  VarLabel::destroy(d_vVel_mmLinSrcLabel);
  VarLabel::destroy(d_vVel_mmLinSrc_FCZLabel);
  VarLabel::destroy(d_vVel_mmLinSrc_FCXLabel);
  VarLabel::destroy(d_vVel_mmNonlinSrc_CCLabel);
  VarLabel::destroy(d_vVel_mmNonlinSrc_CC_CollectLabel);
  VarLabel::destroy(d_vVel_mmNonlinSrcLabel);
  VarLabel::destroy(d_vVel_mmNonlinSrc_FCZLabel);
  VarLabel::destroy(d_vVel_mmNonlinSrc_FCXLabel);

  VarLabel::destroy(d_wVel_mmLinSrc_CCLabel);
  VarLabel::destroy(d_wVel_mmLinSrc_CC_CollectLabel);
  VarLabel::destroy(d_wVel_mmLinSrcLabel);
  VarLabel::destroy(d_wVel_mmLinSrc_FCXLabel);
  VarLabel::destroy(d_wVel_mmLinSrc_FCYLabel);
  VarLabel::destroy(d_wVel_mmNonlinSrc_CCLabel);
  VarLabel::destroy(d_wVel_mmNonlinSrc_CC_CollectLabel);
  VarLabel::destroy(d_wVel_mmNonlinSrcLabel);
  VarLabel::destroy(d_wVel_mmNonlinSrc_FCXLabel);
  VarLabel::destroy(d_wVel_mmNonlinSrc_FCYLabel);
}
