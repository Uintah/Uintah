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


#ifndef Uintah_Component_MPMArchesLabel_h
#define Uintah_Component_MPMArchesLabel_h
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {

    class MPMArchesLabel {
    public:

      MPMArchesLabel();
      ~MPMArchesLabel();

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* solid_fraction_CCLabel;
      const VarLabel* solid_fractionNew_CCLabel;

      // velocity labels for solid materials

      const VarLabel* vel_CCLabel;
      const VarLabel* xvel_CCLabel;
      const VarLabel* yvel_CCLabel;
      const VarLabel* zvel_CCLabel;

      const VarLabel* xvel_FCXLabel;
      const VarLabel* xvel_FCYLabel;
      const VarLabel* xvel_FCZLabel;

      const VarLabel* yvel_FCXLabel;
      const VarLabel* yvel_FCYLabel;
      const VarLabel* yvel_FCZLabel;

      const VarLabel* zvel_FCXLabel;
      const VarLabel* zvel_FCYLabel;
      const VarLabel* zvel_FCZLabel;

      // temperature labels for solid materials

      const VarLabel* tempSolid_CCLabel;
      const VarLabel* tempSolid_FCXLabel;
      const VarLabel* tempSolid_FCYLabel;
      const VarLabel* tempSolid_FCZLabel;

      // reqd by MPM

      const VarLabel* DragForceX_CCLabel;
      const VarLabel* DragForceY_CCLabel;
      const VarLabel* DragForceZ_CCLabel;

      const VarLabel* DragForceY_FCXLabel; 
      const VarLabel* DragForceZ_FCXLabel; 

      const VarLabel* DragForceX_FCYLabel; 
      const VarLabel* DragForceZ_FCYLabel; 

      const VarLabel* DragForceX_FCZLabel; 
      const VarLabel* DragForceY_FCZLabel; 

      const VarLabel* PressureForce_FCXLabel; 
      const VarLabel* PressureForce_FCYLabel; 
      const VarLabel* PressureForce_FCZLabel; 

      const VarLabel* heaTranSolid_CCLabel;
      const VarLabel* heaTranSolid_tmp_CCLabel;
      const VarLabel* heaTranSolid_FCXLabel;
      const VarLabel* heaTranSolid_FCYLabel;
      const VarLabel* heaTranSolid_FCZLabel;
      const VarLabel* heaTranSolid_FCX_RadLabel;
      const VarLabel* heaTranSolid_FCY_RadLabel;
      const VarLabel* heaTranSolid_FCZ_RadLabel;
      const VarLabel* heaTranSolid_NCLabel;

      const VarLabel* SumAllForcesCCLabel; 
      const VarLabel* AccArchesCCLabel; 
      const VarLabel* SumAllForcesNCLabel; 
      const VarLabel* AccArchesNCLabel; 

      // Integrated Solid Property Labels (over 
      // all materials) for Use by Arches

      const VarLabel* integTemp_CCLabel;
      const VarLabel* integHTS_CCLabel;
      const VarLabel* integHTS_FCXLabel;
      const VarLabel* integHTS_FCYLabel;
      const VarLabel* integHTS_FCZLabel;
      const VarLabel* totHT_CCLabel;
      const VarLabel* totHT_FCXLabel;
      const VarLabel* totHT_FCYLabel;
      const VarLabel* totHT_FCZLabel;

      // Heat Flux Labels

      const VarLabel* htfluxConvXLabel;
      const VarLabel* htfluxRadXLabel;
      const VarLabel* htfluxXLabel;
      const VarLabel* htfluxConvYLabel;
      const VarLabel* htfluxRadYLabel;
      const VarLabel* htfluxYLabel;
      const VarLabel* htfluxConvZLabel;
      const VarLabel* htfluxRadZLabel;
      const VarLabel* htfluxZLabel;
      const VarLabel* htfluxConvCCLabel;

      const VarLabel* totHtFluxXLabel;
      const VarLabel* totHtFluxYLabel;
      const VarLabel* totHtFluxZLabel;

      // reqd by Arches

      const VarLabel* void_frac_CCLabel;
      const VarLabel* void_frac_old_CCLabel;
      const VarLabel* void_frac_MPM_CCLabel;
      const VarLabel* void_frac_CutCell_CCLabel;
      const VarLabel* solid_frac_sum_CCLabel;
      const VarLabel* mmCellType_MPMLabel;
      const VarLabel* mmCellType_CutCellLabel;

      // Stability Factor Labels
      
      const VarLabel* KStabilityULabel;
      const VarLabel* KStabilityVLabel;
      const VarLabel* KStabilityWLabel;
      const VarLabel* KStabilityHLabel;

      // reqd by momentum eqns:
      // u-momentum equation source labels

      const VarLabel* d_uVel_mmLinSrc_CCLabel;
      const VarLabel* d_uVel_mmLinSrc_CC_CollectLabel;
      const VarLabel* d_uVel_mmLinSrcLabel;
      const VarLabel* d_uVel_mmLinSrc_FCYLabel;
      const VarLabel* d_uVel_mmLinSrc_FCZLabel;

      const VarLabel* d_uVel_mmNonlinSrc_CCLabel;
      const VarLabel* d_uVel_mmNonlinSrc_CC_CollectLabel;
      const VarLabel* d_uVel_mmNonlinSrcLabel;
      const VarLabel* d_uVel_mmNonlinSrc_FCYLabel;
      const VarLabel* d_uVel_mmNonlinSrc_FCZLabel;

      // v-momentum equation source labels

      const VarLabel* d_vVel_mmLinSrc_CCLabel;
      const VarLabel* d_vVel_mmLinSrc_CC_CollectLabel;
      const VarLabel* d_vVel_mmLinSrcLabel;
      const VarLabel* d_vVel_mmLinSrc_FCZLabel;
      const VarLabel* d_vVel_mmLinSrc_FCXLabel;

      const VarLabel* d_vVel_mmNonlinSrc_CCLabel;
      const VarLabel* d_vVel_mmNonlinSrc_CC_CollectLabel;
      const VarLabel* d_vVel_mmNonlinSrcLabel;
      const VarLabel* d_vVel_mmNonlinSrc_FCZLabel;
      const VarLabel* d_vVel_mmNonlinSrc_FCXLabel;

      // w-momentum equation source labels

      const VarLabel* d_wVel_mmLinSrc_CCLabel;
      const VarLabel* d_wVel_mmLinSrc_CC_CollectLabel;
      const VarLabel* d_wVel_mmLinSrcLabel;
      const VarLabel* d_wVel_mmLinSrc_FCXLabel;
      const VarLabel* d_wVel_mmLinSrc_FCYLabel;

      const VarLabel* d_wVel_mmNonlinSrc_CCLabel;
      const VarLabel* d_wVel_mmNonlinSrc_CC_CollectLabel;
      const VarLabel* d_wVel_mmNonlinSrcLabel;
      const VarLabel* d_wVel_mmNonlinSrc_FCXLabel;
      const VarLabel* d_wVel_mmNonlinSrc_FCYLabel;

      // enthalpy source term labels

      const VarLabel* d_enth_mmLinSrc_tmp_CCLabel;
      const VarLabel* d_enth_mmLinSrc_FCXLabel;
      const VarLabel* d_enth_mmLinSrc_FCYLabel;
      const VarLabel* d_enth_mmLinSrc_FCZLabel;
      const VarLabel* d_enth_mmLinSrc_CCLabel;

      const VarLabel* d_enth_mmNonLinSrc_tmp_CCLabel;
      const VarLabel* d_enth_mmNonLinSrc_FCXLabel;
      const VarLabel* d_enth_mmNonLinSrc_FCYLabel;
      const VarLabel* d_enth_mmNonLinSrc_FCZLabel;
      const VarLabel* d_enth_mmNonLinSrc_CCLabel;

      // cut cell labels;

      const VarLabel* cutCellLabel;
      const VarLabel* d_cutCellInfoLabel;
      const VarLabel* d_normal1Label;
      const VarLabel* d_normal2Label;
      const VarLabel* d_normal3Label;
      const VarLabel* d_normalLabel;
      const VarLabel* d_centroid1Label;
      const VarLabel* d_centroid2Label;
      const VarLabel* d_centroid3Label;
      const VarLabel* d_totAreaLabel;
      const VarLabel* d_pGasAreaFracXPLabel;
      const VarLabel* d_pGasAreaFracXELabel;
      const VarLabel* d_pGasAreaFracYPLabel;
      const VarLabel* d_pGasAreaFracYNLabel;
      const VarLabel* d_pGasAreaFracZPLabel;
      const VarLabel* d_pGasAreaFracZTLabel;
      const VarLabel* d_nextCutCellILabel;
      const VarLabel* d_nextCutCellJLabel;
      const VarLabel* d_nextCutCellKLabel;
      const VarLabel* d_nextWallILabel;
      const VarLabel* d_nextWallJLabel;
      const VarLabel* d_nextWallKLabel;

    };

} // end namespace Uintah

#endif


