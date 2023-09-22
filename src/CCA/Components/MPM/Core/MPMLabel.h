/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_MPMLABEL_H
#define UINTAH_HOMEBREW_MPMLABEL_H

#include <vector>

namespace Uintah {

  class MPMDiffusionLabel;
  class VarLabel;

    class MPMLabel {
    public:

      MPMLabel();
      ~MPMLabel();

      const VarLabel* timeStepLabel;
      const VarLabel* simulationTimeLabel;
      const VarLabel* delTLabel;
      
      // Label containing subclasses.
      MPMDiffusionLabel* diffusion;

      // Label to denote that all particles have been updated, and we're ready
      //   for the final particle update.
      const VarLabel* fAllParticlesUpdated;

      const VarLabel* partCountLabel;
      
      // Heat flux from fire
      const VarLabel* heatRate_CCLabel;

      //non PermanentParticleState
      const VarLabel* pPressureLabel;
      const VarLabel* pScratchVecLabel;
      const VarLabel* pScratchLabel;
      const VarLabel* pVolumeDeformedLabel;
      const VarLabel* TotalVolumeDeformedLabel;

      //PermanentParticleState
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pVelGradLabel;
      const VarLabel* pDeformationMeasureLabel_preReloc;
      const VarLabel* pVelGradLabel_preReloc;
      const VarLabel* pStressLabel;
      const VarLabel* pStressLabel_preReloc;
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pVelocityLabel;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pVelocitySSPlusLabel;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pExternalForceCorner1Label;
      const VarLabel* pExternalForceCorner2Label;
      const VarLabel* pExternalForceCorner3Label;
      const VarLabel* pExternalForceCorner4Label;
      const VarLabel* pExtForceLabel_preReloc;
      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pSurfLabel;
      const VarLabel* pSurfLabel_preReloc;
      const VarLabel* pSurfGradLabel;
      const VarLabel* pSurfGradLabel_preReloc;
      const VarLabel* pLastLevelLabel;
      const VarLabel* pLastLevelLabel_preReloc;
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureLabel_preReloc; //for heat conduction
      const VarLabel* pTempCurrentLabel; //for thermal stress 
      const VarLabel* pTempPreviousLabel; //for thermal stress 
      const VarLabel* pTempPreviousLabel_preReloc; //for thermal stress  
      const VarLabel* pdTdtLabel; //for heat conduction
      const VarLabel* pdTdtLabel_preReloc; //for heat conduction
      const VarLabel* pExternalHeatRateLabel; //for heat conduction
      const VarLabel* pExternalHeatRateLabel_preReloc; //for heat conduction
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;
      const VarLabel* pSizeLabel;
      const VarLabel* pSizeLabel_preReloc;
      const VarLabel* pCurSizeLabel;
      const VarLabel* pLocalizedMPMLabel;
      const VarLabel* pLocalizedMPMLabel_preReloc;
      const VarLabel* pRefinedLabel;
      const VarLabel* pRefinedLabel_preReloc;
      const VarLabel* pFiberDirLabel;
      const VarLabel* pFiberDirLabel_preReloc;
      const VarLabel* pScaleFactorLabel;
      const VarLabel* pScaleFactorLabel_preReloc;
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      const VarLabel* pTemperatureGradientLabel_preReloc; //for heat conduction
      
      const VarLabel* gColorLabel;
      const VarLabel* gMassLabel;
      const VarLabel* gMassAllLabel;
      const VarLabel* gMassF0Label;
      const VarLabel* gMassF1Label;
      const VarLabel* gVelocityF0Label;
      const VarLabel* gVelocityF1Label;
      const VarLabel* gInternalForceF0Label;
      const VarLabel* gInternalForceF1Label;
      const VarLabel* gExternalForceF0Label;
      const VarLabel* gExternalForceF1Label;
      const VarLabel* gVelocityStarF0Label;
      const VarLabel* gVelocityStarF1Label;
      const VarLabel* gAccelerationF0Label;
      const VarLabel* gAccelerationF1Label;
      const VarLabel* gAccelerationLabel;
      const VarLabel* gVelocityLabel;
      const VarLabel* gVelocityBCLabel;
      const VarLabel* gVelSPSSPLabel;
      const VarLabel* gVelocityStarLabel;
      const VarLabel* gMatlProminenceLabel;
      const VarLabel* gAlphaMaterialLabel;
      const VarLabel* gNormAlphaToBetaLabel;
      const VarLabel* gPositionLabel;
      const VarLabel* gPositionF0Label;
      const VarLabel* gPositionF1Label;
      const VarLabel* gExternalForceLabel;
      const VarLabel* NC_CCweightLabel;
      const VarLabel* gInternalForceLabel;
      const VarLabel* gTemperatureRateLabel; //for heat conduction
      const VarLabel* gTemperatureLabel; //for heat conduction
      const VarLabel* gSp_volLabel;          // specific volume 
      const VarLabel* gSp_vol_srcLabel;      // specific volume 
      const VarLabel* gTemperatureNoBCLabel; //for heat conduction
      const VarLabel* gTemperatureStarLabel; //for heat conduction
      const VarLabel* gdTdtLabel;
      const VarLabel* gHeatFluxLabel;
      const VarLabel* gExternalHeatRateLabel;
      const VarLabel* gExternalHeatFluxLabel;
      const VarLabel* gHydrostaticStressLabel;
      const VarLabel* gThermalContactTemperatureRateLabel;
      const VarLabel* gNormTractionLabel;
      const VarLabel* gNormTractionF0Label;
      const VarLabel* gNormTractionF1Label;
      const VarLabel* gSurfNormLabel;
      const VarLabel* gSurfNormF0Label;
      const VarLabel* gSurfNormF1Label;
      const VarLabel* gSurfLabel;
      const VarLabel* gSurfGradLabel;
      const VarLabel* gStressLabel;
      const VarLabel* gStressF0Label;
      const VarLabel* gStressF1Label;
      const VarLabel* gStressForSavingLabel;
      const VarLabel* gVolumeLabel;
      const VarLabel* gVolumeF0Label;
      const VarLabel* gVolumeF1Label;
      const VarLabel* cVolumeLabel;
      const VarLabel* numLocInCellLabel;
      const VarLabel* numInCellLabel;
      const VarLabel* gradPAccNCLabel;
      const VarLabel* dTdt_NCLabel; //for heat conduction
      const VarLabel* massBurnFractionLabel; //for burn modeling
      const VarLabel* frictionalWorkLabel;
      const VarLabel* gNumNearParticlesLabel;

      const VarLabel* StrainEnergyLabel;
      const VarLabel* AccStrainEnergyLabel;
      const VarLabel* KineticEnergyLabel;
      const VarLabel* ThermalEnergyLabel;
      const VarLabel* TotalMassLabel;
      const VarLabel* TotalMomentOfInertiaLabel;
      const VarLabel* NeedAddMPMMaterialLabel;
      const VarLabel* BndyForceLabel[6];
      const VarLabel* BndyTractionLabel[6];
      const VarLabel* BndyContactAreaLabel[6];
      const VarLabel* BndyContactCellAreaLabel[6];
      const VarLabel* CenterOfMassPositionLabel;
            VarLabel* SumTransmittedForceLabel; // not a const since we need to modify it.
            VarLabel* SumTransmittedTorqueLabel;// not a const since we need to modify it.
      const VarLabel* TotalMomentumLabel;
      const VarLabel* RigidReactionForceLabel;
      const VarLabel* RigidReactionTorqueLabel;
      const VarLabel* TotalLocalizedParticleLabel;

      const VarLabel* pCellNAPIDLabel;

      // Labels for particle erosion
      const VarLabel* pErosionLabel;
      const VarLabel* pErosionLabel_preReloc;

      // MPM Physical BC labels (permanent particle state)
      const VarLabel* materialPointsPerLoadCurveLabel;
      const VarLabel* pLoadCurveIDLabel;
      const VarLabel* pLoadCurveIDLabel_preReloc;

      const VarLabel* p_qLabel;
      const VarLabel* p_qLabel_preReloc;

      const VarLabel* pDispLabel;
      const VarLabel* pDispLabel_preReloc;
      const VarLabel* gDisplacementLabel;

      // Debugging Labels
      const VarLabel* pColorLabel;
      const VarLabel* pColorLabel_preReloc;

      // Hydro-mechanical coupling
      const VarLabel* ccPorosity;
      const VarLabel* ccPorePressure;
      const VarLabel* ccPorePressureOld;
      const VarLabel* ccRHS_FlowEquation;
      const VarLabel* ccTransmissivityMatrix;
      const VarLabel* pFluidMassLabel;
      const VarLabel* pFluidVelocityLabel;
      const VarLabel* pFluidAccelerationLabel;
      const VarLabel* pSolidMassLabel;
      const VarLabel* pPorosityLabel;
      const VarLabel* pPorosityLabel_preReloc;
      const VarLabel* pPrescribedPorePressureLabel;
      const VarLabel* pPorePressureLabel;
      const VarLabel* pPorePressureFilterLabel;

      const VarLabel* pStressRateLabel;
      const VarLabel* pStressRateLabel_preReloc;
      const VarLabel* gFluidMassBarLabel;
      const VarLabel* gFluidMassLabel;
      const VarLabel* gFluidVelocityLabel;
      const VarLabel* FluidVelInc;
      const VarLabel* gFluidVelocityStarLabel;
      const VarLabel* gFluidAccelerationLabel;
      const VarLabel* gInternalFluidForceLabel;
      const VarLabel* gExternalFluidForceLabel;
      const VarLabel* gInternalDragForceLabel;
      const VarLabel* gFlowInertiaForceLabel;
      const VarLabel* gPorePressureLabel;
      const VarLabel* gPorePressureFilterLabel;

      const VarLabel* pFluidMassLabel_preReloc;
      const VarLabel* pFluidVelocityLabel_preReloc;
      const VarLabel* pFluidAccelerationLabel_preReloc;
      const VarLabel* pSolidMassLabel_preReloc;
      const VarLabel* pPorePressureLabel_preReloc;
      const VarLabel* pPorePressureFilterLabel_preReloc;
      const VarLabel* gFluidMassBarLabel_preReloc;
      const VarLabel* gFluidMassLabel_preReloc;
      const VarLabel* gFluidVelocityLabel_preReloc;
      const VarLabel* gFluidVelocityStarLabel_preReloc;
      const VarLabel* gFluidAccelerationLabel_preReloc;

      // MPM Hydrostatic BC label
      const VarLabel* boundaryPointsPerCellLabel;

    };
} // End namespace Uintah

#endif
