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
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      const VarLabel* pPressureLabel;
      const VarLabel* p_qLabel;
      const VarLabel* pVolumeDeformedLabel;
      
      //PermanentParticleState
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pDeformationMeasureLabel_preReloc;
      const VarLabel* pStressLabel;
      const VarLabel* pStressLabel_preReloc;
      const VarLabel* pStress_veLabel;
      const VarLabel* pStress_eLabel;
      const VarLabel* pStress_ve_vLabel;
      const VarLabel* pStress_ve_dLabel;
      const VarLabel* pStress_e_vLabel;
      const VarLabel* pStress_e_dLabel;
      const VarLabel* pCrackRadiusLabel;
      const VarLabel* pCrackRadiusLabel_preReloc;
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pVelocityLabel;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pExtForceLabel_preReloc;
      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pSurfLabel;
      const VarLabel* pSurfLabel_preReloc;
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureLabel_preReloc; //for heat conduction
      const VarLabel* pSp_volLabel; 
      const VarLabel* pSp_volLabel_preReloc;
      const VarLabel* pExternalHeatRateLabel; //for heat conduction
      const VarLabel* pExternalHeatRateLabel_preReloc; //for heat conduction
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;
      const VarLabel* pSizeLabel;
      const VarLabel* pSizeLabel_preReloc;

      const VarLabel* pTang1Label;
      const VarLabel* pTang1Label_preReloc;
      const VarLabel* pTang2Label;
      const VarLabel* pTang2Label_preReloc;
      const VarLabel* pNormLabel;
      const VarLabel* pNormLabel_preReloc;
      
      const VarLabel* gMassLabel;
      const VarLabel* gAccelerationLabel;
      const VarLabel* gMomExedAccelerationLabel;
      const VarLabel* gVelocityLabel;
      const VarLabel* gMomExedVelocityLabel;
      const VarLabel* gVelocityStarLabel;
      const VarLabel* gMomExedVelocityStarLabel;
      const VarLabel* gExternalForceLabel;
      const VarLabel* gInternalForceLabel;
      const VarLabel* gContactForceLabel;
      const VarLabel* gContactLabel;
      const VarLabel* gTemperatureRateLabel; //for heat conduction
      const VarLabel* gTemperatureLabel; //for heat conduction
      const VarLabel* gSp_volLabel;          // specific volume 
      const VarLabel* gSp_vol_srcLabel;      // specific volume 
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
      const VarLabel* frictionalWorkLabel;
      const VarLabel* gNumNearParticlesLabel;
      
      const VarLabel* fVelocityLabel; //for interaction with ICE
      const VarLabel* fMassLabel; //for interaction with ICE

      const VarLabel* AccArchesNCLabel; //for interaction with Arches, Fluid Mechanics
      const VarLabel* heaTranSolid_NCLabel; //for interaction with Arches, Heat Transfer

      const VarLabel* StrainEnergyLabel;
      const VarLabel* AccStrainEnergyLabel;
      const VarLabel* KineticEnergyLabel;
      const VarLabel* TotalMassLabel;
      const VarLabel* NTractionZMinusLabel;
      const VarLabel* CenterOfMassPositionLabel;
      const VarLabel* CenterOfMassVelocityLabel;

      const VarLabel* pCellNAPIDLabel;

      vector<vector<const VarLabel* > > d_particleState;
      vector<vector<const VarLabel* > > d_particleState_preReloc;


      // Implicit MPM labels
      const VarLabel* gVelocityOldLabel;
      const VarLabel* pVolumeOldLabel;
      const VarLabel* dispNewLabel;
      const VarLabel* dispIncLabel;
      const VarLabel* pAccelerationLabel;
      const VarLabel* dispIncQNorm0;
      const VarLabel* dispIncNormMax;
      const VarLabel* dispIncQNorm;
      const VarLabel* dispIncNorm;

      const VarLabel* bElBarLabel;
      const VarLabel* bElBarLabel_preReloc;
      const VarLabel* pAccelerationLabel_preReloc;
      const VarLabel* pVolumeOldLabel_preReloc;

      // MPM Physical BC labels (permanent particle state)
      const VarLabel* materialPointsPerLoadCurveLabel;
      const VarLabel* pLoadCurveIDLabel;
      const VarLabel* pLoadCurveIDLabel_preReloc;

      // MPM artificial damping labels (updated after each time step)
      const VarLabel* pDampingRateLabel; // Damping rate summed over particles
      const VarLabel* pDampingCoeffLabel; // Calculated damping coefficient
    };
} // End namespace Uintah

#endif
