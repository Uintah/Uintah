#ifndef Uintah_Component_MPMArchesLabel_h
#define Uintah_Component_MPMArchesLabel_h
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {

    class MPMArchesLabel {
    public:

      MPMArchesLabel();
      ~MPMArchesLabel();

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* solid_fraction_CCLabel;

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
      const VarLabel* heaTranSolid_NCLabel;

      const VarLabel* SumAllForcesCCLabel; 
      const VarLabel* AccArchesCCLabel; 
      const VarLabel* SumAllForcesNCLabel; 
      const VarLabel* AccArchesNCLabel; 

      // reqd by Arches

      const VarLabel* void_frac_CCLabel;

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

    };

} // end namespace Uintah

#endif


