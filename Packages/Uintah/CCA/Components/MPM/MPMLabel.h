#ifndef UINTAH_HOMEBREW_MPMLABEL_H
#define UINTAH_HOMEBREW_MPMLABEL_H


#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <vector>

namespace Uintah {

using std::vector;

    class MPMLabel {
    public:

      MPMLabel();
      ~MPMLabel();

      void registerPermanentParticleState(int i,const VarLabel* l,
					  const VarLabel* lp);

      const VarLabel* delTLabel;
      
      //non PermanentParticleState
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pRotationRateLabel;
      const VarLabel* pVisibilityLabel;
      const VarLabel* pStressReleasedLabel;
      const VarLabel* pIsNewlyBrokenLabel;
      const VarLabel* pStressAfterStrainRateLabel;
      const VarLabel* pStressAfterFractureReleaseLabel;

      const VarLabel* pVelocityAfterUpdateLabel;
      const VarLabel* pVelocityAfterFractureLabel;

      const VarLabel* pStrainEnergyLabel;
      const VarLabel* pNewlyBrokenSurfaceNormalLabel;
      
      const VarLabel* pXXLabel;
      
      //PermanentParticleState
      const VarLabel* pStressLabel;
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeDeformedLabel;
      const VarLabel* pMassLabel;
      const VarLabel* pVelocityLabel;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pXLabel;
      const VarLabel* pSurfLabel;
      const VarLabel* pIsBrokenLabel; //for fracture
      const VarLabel* pCrackNormalLabel; //for fracture
      const VarLabel* pCrackSurfaceContactForceLabel;
      const VarLabel* pTensileStrengthLabel; //for fracture
      const VarLabel* pEnergyReleaseRateLabel; //for fracture
      const VarLabel* pImageVelocityLabel;
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      const VarLabel* pTemperatureRateLabel; //for heat conduction
      const VarLabel* pExternalHeatRateLabel; //for heat conduction
      const VarLabel* pParticleIDLabel;
      const VarLabel* pIsIgnitedLabel; //for burn models
      const VarLabel* pMassRateLabel; //for burn models

      const VarLabel* pDeformationMeasureLabel_preReloc;
      const VarLabel* pStressLabel_preReloc;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pExternalForceLabel_preReloc;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pSurfLabel_preReloc;
      const VarLabel* pIsBrokenLabel_preReloc; //for fracture
      const VarLabel* pCrackNormalLabel_preReloc; //for fracture
      const VarLabel* pCrackSurfaceContactForceLabel_preReloc;
      const VarLabel* pTensileStrengthLabel_preReloc; //for fracture
      const VarLabel* pEnergyReleaseRateLabel_preReloc; //for fracture
      const VarLabel* pImageVelocityLabel_preReloc;
      const VarLabel* pTemperatureLabel_preReloc; //for heat conduction
      const VarLabel* pTemperatureGradientLabel_preReloc; //for heat conduction
      const VarLabel* pTemperatureRateLabel_preReloc; //for heat conduction
      const VarLabel* pExternalHeatRateLabel_preReloc; //for heat conduction
      const VarLabel* pParticleIDLabel_preReloc;
      const VarLabel* pIsIgnitedLabel_preReloc; //for burn models
      const VarLabel* pMassRateLabel_preReloc; //for burn models
      
      const VarLabel* gMassLabel;
      const VarLabel* gAccelerationLabel;
      const VarLabel* gMomExedAccelerationLabel;
      const VarLabel* gVelocityLabel;
      const VarLabel* gMomExedVelocityLabel;
      const VarLabel* gVelocityStarLabel;
      const VarLabel* gMomExedVelocityStarLabel;
      const VarLabel* gExternalForceLabel;
      const VarLabel* gInternalForceLabel;
      const VarLabel* gSelfContactLabel; //for fracture
      const VarLabel* gCrackNormalLabel;
      const VarLabel* gTensileStrengthLabel;
      const VarLabel* gTemperatureRateLabel; //for heat conduction
      const VarLabel* gTemperatureLabel; //for heat conduction
      const VarLabel* gTemperatureStarLabel; //for heat conduction
      const VarLabel* gInternalHeatRateLabel;
      const VarLabel* gExternalHeatRateLabel;
      const VarLabel* gThermalContactHeatExchangeRateLabel;
      const VarLabel* gNormTractionLabel;
      const VarLabel* gSurfNormLabel;
      const VarLabel* gStressLabel;
      const VarLabel* gStressForSavingLabel;
      const VarLabel* gVolumeLabel; //for heat conduction
      const VarLabel* gWeightLabel; //for who knows what?
      
      const VarLabel* cBurnedMassLabel; //for burn models

      const VarLabel* fVelocityLabel; //for interaction with ICE
      const VarLabel* fMassLabel; //for interaction with ICE

      const VarLabel* StrainEnergyLabel;
      const VarLabel* KineticEnergyLabel;
      const VarLabel* TotalMassLabel;
      const VarLabel* NTractionZMinusLabel;
      const VarLabel* CenterOfMassPositionLabel;
      const VarLabel* CenterOfMassVelocityLabel;

      const VarLabel* ppNAPIDLabel;

      vector<vector<const VarLabel* > > d_particleState;
      vector<vector<const VarLabel* > > d_particleState_preReloc;
    };
} // End namespace Uintah

#endif
