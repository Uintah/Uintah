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
      const VarLabel* d_enthalpyRes;

      const VarLabel* d_densityCPLabel;
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

      // Pressure Coeff Labels
      // for new_dw in pressuresolver::linearizeMatrix
      const VarLabel* d_presCoefPBLMLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // U-Velocity Labels

      // for old_dw in computePressureBC and for new_dw in linearsolve
      const VarLabel* d_uVelocitySPBCLabel;



      // V-Velocity Labels

      // for old_dw in computePressureBC
      const VarLabel* d_vVelocitySPBCLabel;


      // W-Velocity Labels

      // for old_dw in computePressureBC
      const VarLabel* d_wVelocitySPBCLabel;


      // Scalar Labels
      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess

      const VarLabel* d_scalarSPLabel;
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

       // Labels for Thermal NOx

      const VarLabel* d_thermalnoxSPLabel;
      const VarLabel* d_thermalnoxTempLabel;

      // thermal NOx Coef

      const VarLabel* d_thermalnoxCoefSBLMLabel;

      // thermal NOx Diffusion Coef

      const VarLabel* d_thermalnoxDiffCoefLabel;

      // thermal NOx NonLinear Src

      const VarLabel* d_thermalnoxNonLinSrcSBLMLabel;

      // thermal NOx source term from properties
      const VarLabel* d_thermalnoxSRCINLabel;

      // End of Thermal NOx labels


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
      const VarLabel* d_LIJCompLabel;

      const VarLabel* d_scalarFluxCompLabel;
      
      // labels for dynamic procedure
      const VarLabel* d_strainMagnitudeLabel;
      const VarLabel* d_strainMagnitudeMLLabel;
      const VarLabel* d_strainMagnitudeMMLabel;
      const VarLabel* d_lalphaLabel;
      const VarLabel* d_cbetaHATalphaLabel;
      const VarLabel* d_alphaalphaLabel;
      const VarLabel* d_CsLabel;
      const VarLabel* d_sumUUULabel;
      const VarLabel* d_sumDllLabel;
      const VarLabel* d_sumSijSijLabel;
      const VarLabel* d_sumDllMinusLabel;
      const VarLabel* d_sumPointsLabel;
      const VarLabel* d_sumUUUPredLabel;
      const VarLabel* d_sumDllPredLabel;
      const VarLabel* d_sumSijSijPredLabel;
      const VarLabel* d_sumDllMinusPredLabel;
      const VarLabel* d_sumPointsPredLabel;
      const VarLabel* d_sumUUUIntermLabel;
      const VarLabel* d_sumDllIntermLabel;
      const VarLabel* d_sumSijSijIntermLabel;
      const VarLabel* d_sumDllMinusIntermLabel;
      const VarLabel* d_sumPointsIntermLabel;
      //label for odt model
      const VarLabel* d_odtDataLabel;

      // label for ref_density and pressure

      const VarLabel* d_refDensity_label;
      const VarLabel* d_refDensityPred_label;
      const VarLabel* d_refPressure_label;

      // labels for nonlinear residuals

      const VarLabel* d_presResidPSLabel;
      const VarLabel* d_presTruncPSLabel;
      const VarLabel* d_uVelResidPSLabel;
      const VarLabel* d_uVelTruncPSLabel;
      const VarLabel* d_vVelResidPSLabel;
      const VarLabel* d_vVelTruncPSLabel;
      const VarLabel* d_wVelResidPSLabel;
      const VarLabel* d_wVelTruncPSLabel;
      const VarLabel* d_scalarResidLabel;
      const VarLabel* d_scalarTruncLabel;
      const VarLabel* d_pressureRes;
      const VarLabel* d_uVelocityRes;
      const VarLabel* d_vVelocityRes;
      const VarLabel* d_wVelocityRes;
      const VarLabel* d_scalarRes;
      const VarLabel* d_reactscalarRes;
      const VarLabel* d_thermalnoxRes;


      // Not sure what these labels are for

      const VarLabel* d_DUPBLMLabel;
      const VarLabel* d_DVPBLMLabel;
      const VarLabel* d_DWPBLMLabel;
      const VarLabel* d_DUMBLMLabel;
      const VarLabel* d_DVMBLMLabel;
      const VarLabel* d_DWMBLMLabel;

      // For storing the interpolated CC Velocity Variables

      const VarLabel* d_oldCCVelocityLabel;
      const VarLabel* d_newCCVelocityLabel;
      const VarLabel* d_newCCVelMagLabel;
      const VarLabel* d_newCCUVelocityLabel;
      const VarLabel* d_newCCVVelocityLabel;
      const VarLabel* d_newCCWVelocityLabel;

      // for pressure grad term in momentum

      const VarLabel* d_pressGradUSuLabel;
      const VarLabel* d_pressGradVSuLabel;
      const VarLabel* d_pressGradWSuLabel;

      // for multimaterial

      const VarLabel* d_mmcellTypeLabel;
      const VarLabel* d_mmgasVolFracLabel;

      // for reacting flows
      const VarLabel* d_tempINLabel;
      const VarLabel* d_cpINLabel;
      const VarLabel* d_co2INLabel;
      const VarLabel* d_h2oINLabel;
      const VarLabel* d_c2h2INLabel;
      const VarLabel* d_h2sINLabel;
      const VarLabel* d_so2INLabel;
      const VarLabel* d_so3INLabel;
      const VarLabel* d_coINLabel;
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
// labels for max(abs(velocity)) for Lax-Friedrichs flux
      const VarLabel* d_maxAbsU_label;
      const VarLabel* d_maxAbsV_label;
      const VarLabel* d_maxAbsW_label;
      const VarLabel* d_maxAbsUPred_label;
      const VarLabel* d_maxAbsVPred_label;
      const VarLabel* d_maxAbsWPred_label;
      const VarLabel* d_maxAbsUInterm_label;
      const VarLabel* d_maxAbsVInterm_label;
      const VarLabel* d_maxAbsWInterm_label;
      const VarLabel* d_maxUxplus_label;
      const VarLabel* d_maxUxplusPred_label;
      const VarLabel* d_maxUxplusInterm_label;
      const VarLabel* d_avUxplus_label;
      const VarLabel* d_avUxplusPred_label;
      const VarLabel* d_avUxplusInterm_label;
// filtered convection terms in momentum eqn
      const VarLabel* d_filteredRhoUjULabel;
      const VarLabel* d_filteredRhoUjVLabel;
      const VarLabel* d_filteredRhoUjWLabel;
// kinetic energy
      const VarLabel* d_kineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyPredLabel;
      const VarLabel* d_totalKineticEnergyIntermLabel;
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
      const VarLabel* d_carbonEfficiencyLabel;
      const VarLabel* d_scalarFlowRateLabel;
      const VarLabel* d_scalarEfficiencyLabel;
      const VarLabel* d_enthalpyFlowRateLabel;
      const VarLabel* d_enthalpyEfficiencyLabel;
      const VarLabel* d_totalRadSrcLabel;
      const VarLabel* d_normTotalRadSrcLabel;

    }; // End class ArchesLabel
} // End namespace Uintah


#endif

