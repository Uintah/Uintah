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
      const VarLabel* pConnectivityLabel;

      const VarLabel* pTouchNormalLabel;
      const VarLabel* pContactNormalLabel;
      
      const VarLabel* pXXLabel;
      
      const VarLabel* pStressAfterFractureReleaseLabel;
      const VarLabel* pStressAfterStrainRateLabel;
      
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
      const VarLabel* pCrackNormal1Label; //for fracture
      const VarLabel* pCrackNormal2Label; //for fracture
      const VarLabel* pCrackNormal3Label; //for fracture
      const VarLabel* pToughnessLabel; //for fracture
      const VarLabel* pEnergyReleaseRateLabel; //for fracture
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
      const VarLabel* pCrackNormal1Label_preReloc; //for fracture
      const VarLabel* pCrackNormal2Label_preReloc; //for fracture
      const VarLabel* pCrackNormal3Label_preReloc; //for fracture
      const VarLabel* pToughnessLabel_preReloc; //for fracture
      const VarLabel* pEnergyReleaseRateLabel_preReloc; //for fracture
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
      const VarLabel* gMassContactLabel; //for crack surface contact
      const VarLabel* gradPressNCLabel;
      const VarLabel* dTdt_NCLabel; //for heat conduction
      
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
