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

      const VarLabel* void_frac_CCLabel; // reqd by Arches
      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* vel_CCLabel;
      // reqd by MPM
      const VarLabel* momExDragForceFCXLabel; 
      const VarLabel* momExDragForceFCYLabel; 
      const VarLabel* momExDragForceFCZLabel; 
      const VarLabel* momExPressureForceFCXLabel; 
      const VarLabel* momExPressureForceFCYLabel; 
      const VarLabel* momExPressureForceFCZLabel; 
      // reqd by momentum eqns for Arches
      const VarLabel* d_uVel_mmLinSrcLabel;
      const VarLabel* d_uVel_mmNonlinSrcLabel;
      const VarLabel* d_vVel_mmLinSrcLabel;
      const VarLabel* d_vVel_mmNonlinSrcLabel;
      const VarLabel* d_wVel_mmLinSrcLabel;
      const VarLabel* d_wVel_mmNonlinSrcLabel;
      // produced by scheduleInterpolateCCToFC
      // if these are duplicates of some of the above, feel free to
      // straighten this out
      const VarLabel* d_xMomFCLabel;
      const VarLabel* d_yMomFCLabel;
      const VarLabel* d_zMomFCLabel;
    };

} // end namespace Uintah

#endif


