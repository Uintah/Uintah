#ifndef UINTAH_HOMEBREW_ANGIOLABEL_H
#define UINTAH_HOMEBREW_ANGIOLABEL_H

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Labels/uintahshare.h>

namespace Uintah {

using std::vector;
  class VarLabel;

    class UINTAHSHARE AngioLabel {
    public:

      AngioLabel();
      ~AngioLabel();

      //PermanentParticleState
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pGrowthLabel;
      const VarLabel* pGrowthLabel_preReloc;
      const VarLabel* pLengthLabel;
      const VarLabel* pLengthLabel_preReloc;
      const VarLabel* pPhiLabel;
      const VarLabel* pPhiLabel_preReloc;
      const VarLabel* pRadiusLabel;
      const VarLabel* pRadiusLabel_preReloc;
      const VarLabel* pTip0Label;
      const VarLabel* pTip0Label_preReloc;
      const VarLabel* pTip1Label;
      const VarLabel* pTip1Label_preReloc;
      const VarLabel* pRecentBranchLabel;
      const VarLabel* pRecentBranchLabel_preReloc;
      const VarLabel* pTimeOfBirthLabel;
      const VarLabel* pTimeOfBirthLabel_preReloc;
      const VarLabel* pParentLabel;
      const VarLabel* pParentLabel_preReloc;
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;

      // Grid Variables
      const VarLabel* VesselDensityLabel;
      const VarLabel* SmoothedVesselDensityLabel;
      const VarLabel* VesselDensityGradientLabel;
      const VarLabel* CollagenThetaLabel;
      const VarLabel* CollagenDevLabel;

      //Miscellaneous Variables
      const VarLabel* delTLabel;
      const VarLabel* partCountLabel;
      const VarLabel* pCellNAPIDLabel;
    };
} // End namespace Uintah

#endif
