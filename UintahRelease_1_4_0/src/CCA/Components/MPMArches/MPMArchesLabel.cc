/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/MPMArches/MPMArches.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/MPMArches/CutCellInfoP.h>
#include <CCA/Components/MPMArches/CutCellInfo.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>

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
  solid_fractionNew_CCLabel = VarLabel::create( "solid_fractionNew",
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

  // solid temperature labels:

  tempSolid_CCLabel =  VarLabel::create("tempSolid_CC",
					CCVariable<double>::getTypeDescription() );
  tempSolid_FCXLabel = VarLabel::create("tempSolid_FCX",
					SFCXVariable<double>::getTypeDescription() );
  tempSolid_FCYLabel = VarLabel::create("tempSolid_FCY",
					SFCYVariable<double>::getTypeDescription() );
  tempSolid_FCZLabel = VarLabel::create("tempSolid_FCZ",
					SFCZVariable<double>::getTypeDescription() );
  // Drag Force labels for solid: CC

  DragForceX_CCLabel = VarLabel::create("DragForceX_CC",
					CCVariable<double>::getTypeDescription() );
  DragForceY_CCLabel = VarLabel::create("DragForceY_CC",
					CCVariable<double>::getTypeDescription() );
  DragForceZ_CCLabel = VarLabel::create("DragForceZ_CC",
					CCVariable<double>::getTypeDescription() );

  // FCX (y- and z-drag forces, acting on x-face)

  DragForceY_FCXLabel = VarLabel::create("DragForceY_FCX",
					 SFCXVariable<double>::getTypeDescription() );
  DragForceZ_FCXLabel = VarLabel::create("DragForceZ_FCX",
					 SFCXVariable<double>::getTypeDescription() );

  // FCY (x- and z-drag forces, acting on y-face)

  DragForceX_FCYLabel = VarLabel::create("DragForceX_FCY",
					 SFCYVariable<double>::getTypeDescription() );
  DragForceZ_FCYLabel = VarLabel::create("DragForceZ_FCY",
					 SFCYVariable<double>::getTypeDescription() );

  // FCZ (x- and y-drag forces, acting on z-face)

  DragForceX_FCZLabel = VarLabel::create("DragForceX_FCZ",
					 SFCZVariable<double>::getTypeDescription() );
  DragForceY_FCZLabel = VarLabel::create("DragForceY_FCZ",
					 SFCZVariable<double>::getTypeDescription() );

  // Pressure Forces:

  PressureForce_FCXLabel = VarLabel::create("PressureForce_FCX",
					    SFCXVariable<double>::getTypeDescription() );
  PressureForce_FCYLabel = VarLabel::create("PressureForce_FCY",
					    SFCYVariable<double>::getTypeDescription() );
  PressureForce_FCZLabel = VarLabel::create("PressureForce_FCZ",
					    SFCZVariable<double>::getTypeDescription() );

  // heat transfer labels for solid

  heaTranSolid_CCLabel = VarLabel::create("heaTranSolid_CC",
					  CCVariable<double>::getTypeDescription() );
  heaTranSolid_tmp_CCLabel = VarLabel::create("heaTranSolid_tmp_CC",
					      CCVariable<double>::getTypeDescription() );
  heaTranSolid_FCXLabel = VarLabel::create("heaTranSolid_FCX",
					   SFCXVariable<double>::getTypeDescription() );
  heaTranSolid_FCYLabel = VarLabel::create("heaTranSolid_FCY",
					   SFCYVariable<double>::getTypeDescription() );
  heaTranSolid_FCZLabel = VarLabel::create("heaTranSolid_FCZ",
					   SFCZVariable<double>::getTypeDescription() );
  heaTranSolid_FCX_RadLabel = VarLabel::create("heaTranSolid_FCX_Rad",
					   SFCXVariable<double>::getTypeDescription() );
  heaTranSolid_FCY_RadLabel = VarLabel::create("heaTranSolid_FCY_Rad",
					   SFCYVariable<double>::getTypeDescription() );
  heaTranSolid_FCZ_RadLabel = VarLabel::create("heaTranSolid_FCZ_Rad",
					   SFCZVariable<double>::getTypeDescription() );
  heaTranSolid_NCLabel = VarLabel::create("heaTranSolid_NC",
					  NCVariable<double>::getTypeDescription() );

  // labels for solid material (MPM) momentum calculations:

  SumAllForcesCCLabel = VarLabel::create("SumAllForcesCC",
					 CCVariable<Vector>::getTypeDescription() );
  AccArchesCCLabel = VarLabel::create("AccArchesCC",
				      CCVariable<Vector>::getTypeDescription() );
  SumAllForcesNCLabel = VarLabel::create("SumAllForcesNC",
					 NCVariable<Vector>::getTypeDescription() );
  AccArchesNCLabel = VarLabel::create("AccArchesNC",
				      NCVariable<Vector>::getTypeDescription() );

  // Integrated Solid Property Labels (over 
  // all materials) for Use by Arches

  integTemp_CCLabel =  VarLabel::create("integTempSolid_CC",
					CCVariable<double>::getTypeDescription() );
  integHTS_CCLabel =  VarLabel::create("integHTSolid_CC",
					CCVariable<double>::getTypeDescription() );
  integHTS_FCXLabel =  VarLabel::create("integHTSolid_FCX",
					SFCXVariable<double>::getTypeDescription() );
  integHTS_FCYLabel =  VarLabel::create("integHTSolid_FCY",
					SFCYVariable<double>::getTypeDescription() );
  integHTS_FCZLabel =  VarLabel::create("integHTSolid_FCZ",
					SFCZVariable<double>::getTypeDescription() );
  totHT_CCLabel = VarLabel::create("totalHTSolid",
				   CCVariable<double>::getTypeDescription() );
  totHT_FCXLabel = VarLabel::create("totalHTSolid_FCX",
				   SFCXVariable<double>::getTypeDescription() );
  totHT_FCYLabel = VarLabel::create("totalHTSolid_FCY",
				   SFCYVariable<double>::getTypeDescription() );
  totHT_FCZLabel = VarLabel::create("totalHTSolid_FCZ",
				   SFCZVariable<double>::getTypeDescription() );

  // Heat Flux Labels

  htfluxConvXLabel = VarLabel::create("htfluxConvX",
				      SFCXVariable<double>::getTypeDescription() );
  htfluxRadXLabel = VarLabel::create("htfluxRadX",
				      SFCXVariable<double>::getTypeDescription() );
  htfluxXLabel = VarLabel::create("htfluxX",
				  SFCXVariable<double>::getTypeDescription() );
  htfluxConvYLabel = VarLabel::create("htfluxConvY",
				      SFCYVariable<double>::getTypeDescription() );
  htfluxRadYLabel = VarLabel::create("htfluxRadY",
				      SFCYVariable<double>::getTypeDescription() );
  htfluxYLabel = VarLabel::create("htfluxY",
				  SFCYVariable<double>::getTypeDescription() );
  htfluxConvZLabel = VarLabel::create("htfluxConvZ",
				      SFCZVariable<double>::getTypeDescription() );
  htfluxRadZLabel = VarLabel::create("htfluxRadZ",
				      SFCZVariable<double>::getTypeDescription() );
  htfluxZLabel = VarLabel::create("htfluxZ",
				  SFCZVariable<double>::getTypeDescription() );
  htfluxConvCCLabel = VarLabel::create("htfluxConvCC",
				       CCVariable<double>::getTypeDescription() );

  totHtFluxXLabel = VarLabel::create("totHtFluxX",
				     SFCXVariable<double>::getTypeDescription() );
  totHtFluxYLabel = VarLabel::create("totHtFluxY",
				     SFCYVariable<double>::getTypeDescription() );
  totHtFluxZLabel = VarLabel::create("totHtFluxZ",
				     SFCZVariable<double>::getTypeDescription() );

  // Labels for gas calculations

  // Gas void fraction and solid fraction sum

  void_frac_CCLabel  = VarLabel::create("void_frac_CC",
					CCVariable<double>::getTypeDescription() );
  void_frac_old_CCLabel  = VarLabel::create("void_frac_CC_old",
					    CCVariable<double>::getTypeDescription() );
  void_frac_MPM_CCLabel  = VarLabel::create("void_frac_MPM_CC",
					    CCVariable<double>::getTypeDescription() );
  void_frac_CutCell_CCLabel  = VarLabel::create("void_frac_CutCell_CC",
						CCVariable<double>::getTypeDescription() );
  solid_frac_sum_CCLabel  = VarLabel::create("solid_frac_sum",
					     CCVariable<double>::getTypeDescription() );
  mmCellType_CutCellLabel  = VarLabel::create("mmCellType_CutCell",
					      CCVariable<int>::getTypeDescription() );
  mmCellType_MPMLabel  = VarLabel::create("mmCellType_MPM",
					  CCVariable<int>::getTypeDescription() );

  // Stability Factor Labels

  KStabilityULabel  = VarLabel::create("KStabilityU",
				       CCVariable<double>::getTypeDescription() );
  KStabilityVLabel  = VarLabel::create("KStabilityV",
				       CCVariable<double>::getTypeDescription() );
  KStabilityWLabel  = VarLabel::create("KStabilityW",
				       CCVariable<double>::getTypeDescription() );
  KStabilityHLabel  = VarLabel::create("KStabilityH",
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

  // enthalpy source term labels

  d_enth_mmLinSrc_tmp_CCLabel = VarLabel::create("enth_mmLinSrc_tmp_CC",
						 CCVariable<double>::getTypeDescription() );
  d_enth_mmLinSrc_FCXLabel = VarLabel::create("enth_mmLinSrc_FCX",
					      SFCXVariable<double>::getTypeDescription() );
  d_enth_mmLinSrc_FCYLabel = VarLabel::create("enth_mmLinSrc_FCY",
					      SFCYVariable<double>::getTypeDescription() );
  d_enth_mmLinSrc_FCZLabel = VarLabel::create("enth_mmLinSrc_FCZ",
					      SFCZVariable<double>::getTypeDescription() );
  d_enth_mmLinSrc_CCLabel = VarLabel::create("enth_mmLinSrc_CC",
					     CCVariable<double>::getTypeDescription() );

  d_enth_mmNonLinSrc_tmp_CCLabel = VarLabel::create("enth_mmNonLinSrc_tmp_CC",
						    CCVariable<double>::getTypeDescription() );
  d_enth_mmNonLinSrc_FCXLabel = VarLabel::create("enth_mmNonLinSrc_FCX",
						 SFCXVariable<double>::getTypeDescription() );
  d_enth_mmNonLinSrc_FCYLabel = VarLabel::create("enth_mmNonLinSrc_FCY",
						 SFCYVariable<double>::getTypeDescription() );
  d_enth_mmNonLinSrc_FCZLabel = VarLabel::create("enth_mmNonLinSrc_FCZ",
						 SFCZVariable<double>::getTypeDescription() );
  d_enth_mmNonLinSrc_CCLabel = VarLabel::create("enth_mmNonLinSrc_CC",
						CCVariable<double>::getTypeDescription() );

  // cut cell labels

  cutCellLabel = VarLabel::create("cutCell",
  				  CCVariable<cutcell>::getTypeDescription());
  d_cutCellInfoLabel = VarLabel::create("cutCellInfo", 
					PerPatch<CutCellInfoP>::getTypeDescription());
  /*
    Stuff below (upto pGasAreaFracZT) should be removed when
    we use the d_cutcell struct
   */
  d_normal1Label = VarLabel::create("normal1",
				    CCVariable<double>::getTypeDescription() );
  d_normal2Label = VarLabel::create("normal2",
				    CCVariable<double>::getTypeDescription() );
  d_normal3Label = VarLabel::create("normal3",
				    CCVariable<double>::getTypeDescription() );
  d_normalLabel = VarLabel::create("normal",
				   CCVariable<Vector>::getTypeDescription() );

  d_centroid1Label = VarLabel::create("centroid1",
				      CCVariable<double>::getTypeDescription() );
  d_centroid2Label = VarLabel::create("centroid2",
				      CCVariable<double>::getTypeDescription() );
  d_centroid3Label = VarLabel::create("centroid3",
				      CCVariable<double>::getTypeDescription() );
  d_totAreaLabel = VarLabel::create("totArea",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracXPLabel = VarLabel::create("pGasAreaFracXP",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracXELabel = VarLabel::create("pGasAreaFracXE",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracYPLabel = VarLabel::create("pGasAreaFracYP",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracYNLabel = VarLabel::create("pGasAreaFracYN",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracZPLabel = VarLabel::create("pGasAreaFracZP",
				    CCVariable<double>::getTypeDescription() );
  d_pGasAreaFracZTLabel = VarLabel::create("pGasAreaFracZT",
				    CCVariable<double>::getTypeDescription() );
  /*
  */
  d_nextCutCellILabel = VarLabel::create("nextCutCellI", 
				    CCVariable<int>::getTypeDescription() );
  d_nextCutCellJLabel = VarLabel::create("nextCutCellJ", 
				    CCVariable<int>::getTypeDescription() );
  d_nextCutCellKLabel = VarLabel::create("nextCutCellK", 
				    CCVariable<int>::getTypeDescription() );

  d_nextWallILabel = VarLabel::create("nextWallI", 
				    CCVariable<int>::getTypeDescription() );
  d_nextWallJLabel = VarLabel::create("nextWallJ", 
				    CCVariable<int>::getTypeDescription() );
  d_nextWallKLabel = VarLabel::create("nextWallK", 
				    CCVariable<int>::getTypeDescription() );

} 

MPMArchesLabel::~MPMArchesLabel()
{
  VarLabel::destroy(cMassLabel);
  VarLabel::destroy(cVolumeLabel);
  VarLabel::destroy(solid_fraction_CCLabel);
  VarLabel::destroy(solid_fractionNew_CCLabel);

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

  VarLabel::destroy(tempSolid_CCLabel);
  VarLabel::destroy(tempSolid_FCXLabel);
  VarLabel::destroy(tempSolid_FCYLabel);
  VarLabel::destroy(tempSolid_FCZLabel);

  VarLabel::destroy(DragForceX_CCLabel);
  VarLabel::destroy(DragForceY_CCLabel);
  VarLabel::destroy(DragForceZ_CCLabel);

  VarLabel::destroy(DragForceY_FCXLabel);
  VarLabel::destroy(DragForceZ_FCXLabel);

  VarLabel::destroy(DragForceX_FCYLabel);
  VarLabel::destroy(DragForceZ_FCYLabel);

  VarLabel::destroy(DragForceX_FCZLabel);
  VarLabel::destroy(DragForceY_FCZLabel);

  VarLabel::destroy(PressureForce_FCXLabel);
  VarLabel::destroy(PressureForce_FCYLabel);
  VarLabel::destroy(PressureForce_FCZLabel);

  VarLabel::destroy(heaTranSolid_CCLabel);
  VarLabel::destroy(heaTranSolid_tmp_CCLabel);
  VarLabel::destroy(heaTranSolid_FCXLabel);
  VarLabel::destroy(heaTranSolid_FCYLabel);
  VarLabel::destroy(heaTranSolid_FCZLabel);
  VarLabel::destroy(heaTranSolid_FCX_RadLabel);
  VarLabel::destroy(heaTranSolid_FCY_RadLabel);
  VarLabel::destroy(heaTranSolid_FCZ_RadLabel);
  VarLabel::destroy(heaTranSolid_NCLabel);

  VarLabel::destroy(SumAllForcesCCLabel);
  VarLabel::destroy(SumAllForcesNCLabel);
  VarLabel::destroy(AccArchesCCLabel);
  VarLabel::destroy(AccArchesNCLabel);

  VarLabel::destroy(integTemp_CCLabel);
  VarLabel::destroy(integHTS_CCLabel);
  VarLabel::destroy(integHTS_FCXLabel);
  VarLabel::destroy(integHTS_FCYLabel);
  VarLabel::destroy(integHTS_FCZLabel);
  VarLabel::destroy(totHT_CCLabel);  
  VarLabel::destroy(totHT_FCXLabel);  
  VarLabel::destroy(totHT_FCYLabel);  
  VarLabel::destroy(totHT_FCZLabel);  

  VarLabel::destroy(htfluxConvXLabel);  
  VarLabel::destroy(htfluxRadXLabel);  
  VarLabel::destroy(htfluxXLabel);  
  VarLabel::destroy(htfluxConvYLabel);  
  VarLabel::destroy(htfluxRadYLabel);  
  VarLabel::destroy(htfluxYLabel);  
  VarLabel::destroy(htfluxConvZLabel);  
  VarLabel::destroy(htfluxRadZLabel);  
  VarLabel::destroy(htfluxZLabel);  
  VarLabel::destroy(htfluxConvCCLabel);  

  VarLabel::destroy(totHtFluxXLabel);
  VarLabel::destroy(totHtFluxYLabel);
  VarLabel::destroy(totHtFluxZLabel);

  VarLabel::destroy(void_frac_CCLabel);
  VarLabel::destroy(void_frac_old_CCLabel);
  VarLabel::destroy(void_frac_MPM_CCLabel);
  VarLabel::destroy(void_frac_CutCell_CCLabel);
  VarLabel::destroy(solid_frac_sum_CCLabel);
  VarLabel::destroy(mmCellType_CutCellLabel);
  VarLabel::destroy(mmCellType_MPMLabel);

  VarLabel::destroy(KStabilityULabel);
  VarLabel::destroy(KStabilityVLabel);
  VarLabel::destroy(KStabilityWLabel);
  VarLabel::destroy(KStabilityHLabel);

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

  VarLabel::destroy( d_enth_mmLinSrc_tmp_CCLabel);
  VarLabel::destroy( d_enth_mmLinSrc_FCXLabel);
  VarLabel::destroy( d_enth_mmLinSrc_FCYLabel);
  VarLabel::destroy( d_enth_mmLinSrc_FCZLabel);
  VarLabel::destroy( d_enth_mmLinSrc_CCLabel);

  VarLabel::destroy(d_enth_mmNonLinSrc_tmp_CCLabel);
  VarLabel::destroy(d_enth_mmNonLinSrc_FCXLabel);
  VarLabel::destroy(d_enth_mmNonLinSrc_FCYLabel);
  VarLabel::destroy(d_enth_mmNonLinSrc_FCZLabel);
  VarLabel::destroy(d_enth_mmNonLinSrc_CCLabel);

  VarLabel::destroy(cutCellLabel);
  VarLabel::destroy(d_cutCellInfoLabel);
  VarLabel::destroy(d_normal1Label);
  VarLabel::destroy(d_normal2Label);
  VarLabel::destroy(d_normal3Label);
  VarLabel::destroy(d_normalLabel);
  VarLabel::destroy(d_centroid1Label);
  VarLabel::destroy(d_centroid2Label);
  VarLabel::destroy(d_centroid3Label);
  VarLabel::destroy(d_totAreaLabel);

  VarLabel::destroy(d_pGasAreaFracXPLabel);
  VarLabel::destroy(d_pGasAreaFracXELabel);
  VarLabel::destroy(d_pGasAreaFracYPLabel);
  VarLabel::destroy(d_pGasAreaFracYNLabel);
  VarLabel::destroy(d_pGasAreaFracZPLabel);
  VarLabel::destroy(d_pGasAreaFracZTLabel);

  VarLabel::destroy(d_nextCutCellILabel);
  VarLabel::destroy(d_nextCutCellJLabel);
  VarLabel::destroy(d_nextCutCellKLabel);

  VarLabel::destroy(d_nextWallILabel);
  VarLabel::destroy(d_nextWallJLabel);
  VarLabel::destroy(d_nextWallKLabel);

}
