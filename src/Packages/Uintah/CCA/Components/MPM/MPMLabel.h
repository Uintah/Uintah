#ifndef UINTAH_HOMEBREW_MPMLABEL_H
#define UINTAH_HOMEBREW_MPMLABEL_H


#include <vector>

namespace Uintah {

using std::vector;
  class VarLabel;

    class MPMLabel {
    public:

      MPMLabel();
      ~MPMLabel();

      void registerPermanentParticleState(int i,const VarLabel* l,
					  const VarLabel* lp);

      const VarLabel* delTLabel;
      const VarLabel* doMechLabel;

      const VarLabel* partCountLabel;
      
      //non PermanentParticleState
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pConnectivityLabel;
      const VarLabel* pCrackEffectiveLabel;
      const VarLabel* pContactForceLabel;

      const VarLabel* pXXLabel;
      
      const VarLabel* pStressLabel_afterFracture;
      const VarLabel* pStressLabel_afterStrainRate;

      const VarLabel* pVelocityLabel_afterFracture;
      const VarLabel* pVelocityLabel_afterUpdate;
      
      const VarLabel* pStrainEnergyLabel;
      const VarLabel* pRotationRateLabel;

      const VarLabel* pPressureLabel;
      
      //PermanentParticleState
      const VarLabel* pStressLabel;
      const VarLabel* pCrackRadiusLabel;
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeDeformedLabel;
      const VarLabel* pMassLabel;
      const VarLabel* pVelocityLabel;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pXLabel;
      const VarLabel* pSurfLabel;
      const VarLabel* pIsBrokenLabel; //for fracture
      const VarLabel* pCrackNormalLabel; //for fracture
      const VarLabel* pTipNormalLabel; //for fracture
      const VarLabel* pExtensionDirectionLabel; //for fracture
      const VarLabel* pToughnessLabel; //for fracture
      const VarLabel* pEnergyReleaseRateLabel; //for fracture
      const VarLabel* pCrackSurfacePressureLabel; //for explosive fracture
      const VarLabel* pDisplacementLabel; //for fracture
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      const VarLabel* pExternalHeatRateLabel; //for heat conduction
      const VarLabel* pParticleIDLabel;
      const VarLabel* pIsIgnitedLabel; //for burn models
      const VarLabel* pMassRateLabel; //for burn models

      const VarLabel* pDeformationMeasureLabel_preReloc;
      const VarLabel* pStressLabel_preReloc;
      const VarLabel* pCrackRadiusLabel_preReloc;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pExternalForceLabel_preReloc;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pSurfLabel_preReloc;
      const VarLabel* pIsBrokenLabel_preReloc; //for fracture
      const VarLabel* pCrackNormalLabel_preReloc; //for fracture
      const VarLabel* pTipNormalLabel_preReloc; //for fracture
      const VarLabel* pExtensionDirectionLabel_preReloc; //for fracture
      const VarLabel* pToughnessLabel_preReloc; //for fracture
      const VarLabel* pEnergyReleaseRateLabel_preReloc; //for fracture
      const VarLabel* pCrackSurfacePressureLabel_preReloc; //for explosive fracture
      const VarLabel* pDisplacementLabel_preReloc; //for fracture
      const VarLabel* pTemperatureLabel_preReloc; //for heat conduction
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
      const VarLabel* gTemperatureRateLabel; //for heat conduction
      const VarLabel* gTemperatureLabel; //for heat conduction
      const VarLabel* gTemperatureNoBCLabel; //for heat conduction
      const VarLabel* gTemperatureStarLabel; //for heat conduction
      const VarLabel* gInternalHeatRateLabel;
      const VarLabel* gExternalHeatRateLabel;
      const VarLabel* gThermalContactHeatExchangeRateLabel;
      const VarLabel* gNormTractionLabel;
      const VarLabel* gSurfNormLabel;
      const VarLabel* gStressLabel;
      const VarLabel* gStressForSavingLabel;
      const VarLabel* gVolumeLabel;
      const VarLabel* gWeightLabel; //for who knows what?
      const VarLabel* gradPAccNCLabel;
      const VarLabel* dTdt_NCLabel; //for heat conduction
      const VarLabel* massBurnFractionLabel; //for burn modeling
      
      const VarLabel* fVelocityLabel; //for interaction with ICE
      const VarLabel* fMassLabel; //for interaction with ICE

      const VarLabel* AccArchesNCLabel;

      const VarLabel* StrainEnergyLabel;
      const VarLabel* KineticEnergyLabel;
      const VarLabel* TotalMassLabel;
      const VarLabel* NTractionZMinusLabel;
      const VarLabel* CenterOfMassPositionLabel;
      const VarLabel* CenterOfMassVelocityLabel;

      const VarLabel* pCellNAPIDLabel;

      vector<vector<const VarLabel* > > d_particleState;
      vector<vector<const VarLabel* > > d_particleState_preReloc;
    };
} // End namespace Uintah

#endif
