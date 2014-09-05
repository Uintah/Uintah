//----- ArchesLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesLabel_h
#define Uintah_Components_Arches_ArchesLabel_h

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Util/Handle.h>


/**************************************
CLASS
   ArchesLabel
   
   Class ArchesLabel creates and stores the VarLabels that are used in Arches

GENERAL INFORMATION
   ArchesLabel.h - declaration of the class
   
   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
   
   Creation Date:   July 18, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

namespace Uintah {
  class VarLabel;
    class ArchesLabel {
    public:

      ArchesLabel();
      ~ArchesLabel();
      void setSharedState(SimulationStateP& sharedState);
     
      SimulationStateP d_sharedState;

      // material subset for stencils
      MaterialSubset* d_stencilMatl;

      MaterialSubset* d_vectorMatl;
      MaterialSubset* d_tensorMatl;
      MaterialSubset* d_symTensorMatl;

      // Cell Information
      // for old_dw, perpatch var

      const VarLabel* d_cellInfoLabel;

      // Cell type

      const VarLabel* d_cellTypeLabel;//computed for old_dw in cellTypeInit
     
      // Labels for inlet and flow rate
      const VarLabel* d_totalflowINLabel;
      const VarLabel* d_totalflowOUTLabel;
      const VarLabel* d_netflowOUTBCLabel;
      // net outlet area, mass balance, and overall outlet velocity
      const VarLabel* d_totalAreaOUTLabel;
      const VarLabel* d_denAccumLabel;
      // Density Labels

      const VarLabel* d_densityCPLabel;
      const VarLabel* d_densityEKTLabel;
      const VarLabel* d_densityGuessLabel;
      const VarLabel* d_densityTempLabel;
      const VarLabel* d_densityOldOldLabel;

      // filtered drhodt
      const VarLabel* d_filterdrhodtLabel;
      // for computing divergence constraint
      const VarLabel* d_drhodfCPLabel;

      // Viscosity Labels
      // for old_dw in computeTurbModel
      const VarLabel* d_viscosityCTSLabel;
      const VarLabel* d_scalarDiffusivityLabel;
      const VarLabel* d_enthalpyDiffusivityLabel;
      const VarLabel* d_reactScalarDiffusivityLabel;


      // Pressure Labels

      // for old_dw in computePressureBC
      const VarLabel* d_pressurePSLabel;
      const VarLabel* d_pressureExtraProjectionLabel;

      // Pressure Coeff Labels
      // for new_dw in pressuresolver::linearizeMatrix
      const VarLabel* d_presCoefPBLMLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // U-Velocity Labels
      const VarLabel* d_uVelocitySPBCLabel;
      // V-Velocity Labels
      const VarLabel* d_vVelocitySPBCLabel;
      // W-Velocity Labels
      const VarLabel* d_wVelocitySPBCLabel;

      const VarLabel* d_uVelocityEKTLabel;
      const VarLabel* d_vVelocityEKTLabel;
      const VarLabel* d_wVelocityEKTLabel;

      // Scalar Labels
      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess

      const VarLabel* d_scalarSPLabel;
      const VarLabel* d_scalarEKTLabel;
      const VarLabel* d_scalarTempLabel;
      const VarLabel* d_scalarFELabel;

      // Scalar variance labels

      // computed for new_dw in Smagorinsky Model

      const VarLabel* d_scalarVarSPLabel;

      // computed for new_dw in Smagorinsky Model for flamelet

      const VarLabel* d_scalarDissSPLabel;

      // Scalar Coef

      const VarLabel* d_scalCoefSBLMLabel;

      // scalar diffusion coeffs, required for divergence constraint
      const VarLabel* d_scalDiffCoefLabel;

      const VarLabel* d_scalDiffCoefSrcLabel;

      const VarLabel* d_enthDiffCoefLabel;

      // Scalar NonLinear Src

      const VarLabel* d_scalNonLinSrcSBLMLabel;


      // reactive scalars


      const VarLabel* d_reactscalarSPLabel;
      const VarLabel* d_reactscalarEKTLabel;
      const VarLabel* d_reactscalarTempLabel;
      const VarLabel* d_reactscalarFELabel;

      // Reactscalar variance labels
      // computed for new_dw in Smagorinsky Model

      const VarLabel* d_reactscalarVarSPLabel;

      // Reactscalar Coef

      const VarLabel* d_reactscalCoefSBLMLabel;

      // Reactscalar Diffusion Coef

      const VarLabel* d_reactscalDiffCoefLabel;
      
      // Reactscalar NonLinear Src

      const VarLabel* d_reactscalNonLinSrcSBLMLabel;

      // labels for scalesimilaritymodels

      const VarLabel* d_stressTensorCompLabel;
      const VarLabel* d_stressSFCXdivLabel;
      const VarLabel* d_stressSFCYdivLabel;
      const VarLabel* d_stressSFCZdivLabel;      
      const VarLabel* d_stressCCXdivLabel;
      const VarLabel* d_stressCCYdivLabel;
      const VarLabel* d_stressCCZdivLabel;
      const VarLabel* d_strainTensorCompLabel;
      const VarLabel* d_betaIJCompLabel;
      const VarLabel* d_cbetaIJCompLabel;
      const VarLabel* d_LIJCompLabel;

      const VarLabel* d_scalarFluxCompLabel;
      
      // labels for dynamic procedure
      const VarLabel* d_strainMagnitudeLabel;
      const VarLabel* d_strainMagnitudeMLLabel;
      const VarLabel* d_strainMagnitudeMMLabel;
      const VarLabel* d_LalphaLabel;
      const VarLabel* d_cbetaHATalphaLabel;
      const VarLabel* d_alphaalphaLabel;
      const VarLabel* d_CsLabel;
      const VarLabel* d_deltaCsLabel;
      //label for odt model
      const VarLabel* d_odtDataLabel;

      // label for ref_density and pressure

      const VarLabel* d_refDensity_label;
      const VarLabel* d_refDensityPred_label;
      const VarLabel* d_refPressure_label;

      // labels for nonlinear residuals



      // For storing the interpolated CC Velocity Variables

      const VarLabel* d_oldCCVelocityLabel;
      const VarLabel* d_newCCVelocityLabel;
      const VarLabel* d_newCCVelMagLabel;
      const VarLabel* d_newCCUVelocityLabel;
      const VarLabel* d_newCCVVelocityLabel;
      const VarLabel* d_newCCWVelocityLabel;

      // for multimaterial

      const VarLabel* d_mmcellTypeLabel;
      const VarLabel* d_mmgasVolFracLabel;

      // for reacting flows
      const VarLabel* d_tempINLabel;
      const VarLabel* d_cpINLabel;
      const VarLabel* d_co2INLabel;
      const VarLabel* d_h2oINLabel;
      const VarLabel* d_normalizedScalarVarLabel;
      const VarLabel* d_heatLossLabel;
      const VarLabel* d_h2sINLabel;
      const VarLabel* d_so2INLabel;
      const VarLabel* d_so3INLabel;
      const VarLabel* d_sulfurINLabel;
      const VarLabel* d_coINLabel;
      const VarLabel* d_c2h2INLabel;
      const VarLabel* d_ch4INLabel;
      const VarLabel* d_denRefArrayLabel;
      const VarLabel* d_densityMicroLabel;
      const VarLabel* d_densityMicroINLabel;
      const VarLabel* d_pressPlusHydroLabel;
      // predicted

      // for outlet bc
      const VarLabel* d_uvwoutLabel;
      // pred-corr labels

      // labels for pressure solver

      const VarLabel* d_uVelRhoHatLabel;
      const VarLabel* d_vVelRhoHatLabel;
      const VarLabel* d_wVelRhoHatLabel;

      const VarLabel* d_uVelRhoHat_CCLabel;
      const VarLabel* d_vVelRhoHat_CCLabel;
      const VarLabel* d_wVelRhoHat_CCLabel;

      // divergence constraint
      const VarLabel* d_divConstraintLabel;

      const VarLabel* d_pressurePredLabel;

      // for enthalpy equation
      const VarLabel* d_enthalpySPLabel;
      const VarLabel* d_enthalpyEKTLabel;
      const VarLabel* d_enthalpyTempLabel;
      const VarLabel* d_enthalpyFELabel;
      // for validation
      const VarLabel* d_enthalpyRXNLabel;


      // Enthalpy Coef

      const VarLabel* d_enthCoefSBLMLabel;

      // Enthalpy NonLinear Src

      const VarLabel* d_enthNonLinSrcSBLMLabel;


      // for radiation
      const VarLabel* d_fvtfiveINLabel;
      const VarLabel* d_tfourINLabel;
      const VarLabel* d_tfiveINLabel;
      const VarLabel* d_tnineINLabel;
      const VarLabel* d_qrgINLabel;
      const VarLabel* d_qrsINLabel;
      const VarLabel* d_absorpINLabel;
      const VarLabel* d_sootFVINLabel;
      const VarLabel* d_abskgINLabel;
      const VarLabel* d_radiationSRCINLabel;
      const VarLabel* d_radiationFluxEINLabel;
      const VarLabel* d_radiationFluxWINLabel;
      const VarLabel* d_radiationFluxNINLabel;
      const VarLabel* d_radiationFluxSINLabel;
      const VarLabel* d_radiationFluxTINLabel;
      const VarLabel* d_radiationFluxBINLabel;
      // reactive scalar source term from properties
      const VarLabel* d_reactscalarSRCINLabel;
      
      

      // runge-kutta 3d order properties labels
      const VarLabel* d_refDensityInterm_label;
      
      // runge-kutta 3d order pressure and momentum labels
      const VarLabel* d_pressureIntermLabel;
      const VarLabel* d_velocityDivergenceLabel;
      const VarLabel* d_vorticityXLabel;
      const VarLabel* d_vorticityYLabel;
      const VarLabel* d_vorticityZLabel;
      const VarLabel* d_vorticityLabel;
      const VarLabel* d_velDivResidualLabel;
      const VarLabel* d_velocityDivergenceBCLabel;
      const VarLabel* d_continuityResidualLabel;

      const VarLabel* d_InitNormLabel;
      const VarLabel* d_ScalarClippedLabel;
      const VarLabel* d_ReactScalarClippedLabel;
      const VarLabel* d_uVelNormLabel;
      const VarLabel* d_vVelNormLabel;
      const VarLabel* d_wVelNormLabel;
      const VarLabel* d_rhoNormLabel;
      const VarLabel* d_negativeDensityGuess_label;
      const VarLabel* d_negativeDensityGuessPred_label;
      const VarLabel* d_negativeDensityGuessInterm_label;
      const VarLabel* d_negativeEKTDensityGuess_label;
      const VarLabel* d_negativeEKTDensityGuessPred_label;
      const VarLabel* d_negativeEKTDensityGuessInterm_label;
      const VarLabel* d_densityLag_label;
      const VarLabel* d_densityLagPred_label;
      const VarLabel* d_densityLagInterm_label;
      const VarLabel* d_densityLagAfterAverage_label;
      const VarLabel* d_densityLagAfterIntermAverage_label;
// kinetic energy
      const VarLabel* d_kineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyPredLabel;
      const VarLabel* d_totalKineticEnergyIntermLabel;
// scalar mms Ln error
      const VarLabel* d_smmsLnErrorLabel;
      const VarLabel* d_totalsmmsLnErrorLabel;
      const VarLabel* d_totalsmmsLnErrorPredLabel;
      const VarLabel* d_totalsmmsLnErrorIntermLabel;
      const VarLabel* d_totalsmmsExactSolLabel;
      const VarLabel* d_totalsmmsExactSolPredLabel;
      const VarLabel* d_totalsmmsExactSolIntermLabel;
// grad P mms Ln error
      const VarLabel* d_gradpmmsLnErrorLabel;
      const VarLabel* d_totalgradpmmsLnErrorLabel;
      const VarLabel* d_totalgradpmmsLnErrorPredLabel;
      const VarLabel* d_totalgradpmmsLnErrorIntermLabel;
      const VarLabel* d_totalgradpmmsExactSolLabel;
      const VarLabel* d_totalgradpmmsExactSolPredLabel;
      const VarLabel* d_totalgradpmmsExactSolIntermLabel;
// u mms Ln error
      const VarLabel* d_ummsLnErrorLabel;
      const VarLabel* d_totalummsLnErrorLabel;
      const VarLabel* d_totalummsLnErrorPredLabel;
      const VarLabel* d_totalummsLnErrorIntermLabel;
      const VarLabel* d_totalummsExactSolLabel;
      const VarLabel* d_totalummsExactSolPredLabel;
      const VarLabel* d_totalummsExactSolIntermLabel;
// v mms Ln error
      const VarLabel* d_vmmsLnErrorLabel;
      const VarLabel* d_totalvmmsLnErrorLabel;
      const VarLabel* d_totalvmmsLnErrorPredLabel;
      const VarLabel* d_totalvmmsLnErrorIntermLabel;
      const VarLabel* d_totalvmmsExactSolLabel;
      const VarLabel* d_totalvmmsExactSolPredLabel;
      const VarLabel* d_totalvmmsExactSolIntermLabel;

// w mms Ln error
      const VarLabel* d_wmmsLnErrorLabel;
      const VarLabel* d_totalwmmsLnErrorLabel;
      const VarLabel* d_totalwmmsLnErrorPredLabel;
      const VarLabel* d_totalwmmsLnErrorIntermLabel;
      const VarLabel* d_totalwmmsExactSolLabel;
      const VarLabel* d_totalwmmsExactSolPredLabel;
      const VarLabel* d_totalwmmsExactSolIntermLabel;

// mass balance labels for RK
      const VarLabel* d_totalflowINPredLabel;
      const VarLabel* d_totalflowOUTPredLabel;
      const VarLabel* d_denAccumPredLabel;
      const VarLabel* d_netflowOUTBCPredLabel;
      const VarLabel* d_totalAreaOUTPredLabel;
      const VarLabel* d_totalflowINIntermLabel;
      const VarLabel* d_totalflowOUTIntermLabel;
      const VarLabel* d_denAccumIntermLabel;
      const VarLabel* d_netflowOUTBCIntermLabel;
      const VarLabel* d_totalAreaOUTIntermLabel;
 
      const VarLabel* d_oldDeltaTLabel;
// test filtered terms for variable density dynamic Smagorinsky model
      const VarLabel* d_filterRhoULabel;
      const VarLabel* d_filterRhoVLabel;
      const VarLabel* d_filterRhoWLabel;
      const VarLabel* d_filterRhoLabel;
      const VarLabel* d_filterRhoFLabel;
      const VarLabel* d_filterRhoELabel;
      const VarLabel* d_filterRhoRFLabel;
      const VarLabel* d_scalarGradientCompLabel;
      const VarLabel* d_filterScalarGradientCompLabel;
      const VarLabel* d_enthalpyGradientCompLabel;
      const VarLabel* d_filterEnthalpyGradientCompLabel;
      const VarLabel* d_reactScalarGradientCompLabel;
      const VarLabel* d_filterReactScalarGradientCompLabel;
      const VarLabel* d_filterStrainTensorCompLabel;
      const VarLabel* d_scalarNumeratorLabel; 
      const VarLabel* d_scalarDenominatorLabel; 
      const VarLabel* d_enthalpyNumeratorLabel; 
      const VarLabel* d_enthalpyDenominatorLabel; 
      const VarLabel* d_reactScalarNumeratorLabel; 
      const VarLabel* d_reactScalarDenominatorLabel; 
      const VarLabel* d_ShFLabel;
      const VarLabel* d_ShELabel;
      const VarLabel* d_ShRFLabel;
      const VarLabel* d_CO2FlowRateLabel;
      const VarLabel* d_SO2FlowRateLabel;
      const VarLabel* d_carbonEfficiencyLabel;
      const VarLabel* d_sulfurEfficiencyLabel;
      const VarLabel* d_scalarFlowRateLabel;
      const VarLabel* d_scalarEfficiencyLabel;
      const VarLabel* d_enthalpyFlowRateLabel;
      const VarLabel* d_enthalpyEfficiencyLabel;
      const VarLabel* d_totalRadSrcLabel;
      const VarLabel* d_normTotalRadSrcLabel;

      //mms force term labels
      const VarLabel* d_uFmmsLabel;
      const VarLabel* d_vFmmsLabel;
      const VarLabel* d_wFmmsLabel;

    }; // End class ArchesLabel
} // End namespace Uintah


#endif

