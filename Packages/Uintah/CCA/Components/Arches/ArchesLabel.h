//----- ArchesLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesLabel_h
#define Uintah_Components_Arches_ArchesLabel_h

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>


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
      // material subset for turbulent stress, required by scalesimilarity model
      MaterialSubset* d_stressTensorMatl;
      MaterialSubset* d_stressSymTensorMatl;
      MaterialSubset* d_scalarFluxMatl;

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
      const VarLabel* d_strainTensorCompLabel;

      const VarLabel* d_scalarFluxCompLabel;
      
      // labels for dynamic procedure
      const VarLabel* d_strainMagnitudeLabel;
      const VarLabel* d_strainMagnitudeMLLabel;
      const VarLabel* d_strainMagnitudeMMLabel;
      const VarLabel* d_CsLabel;
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
      const VarLabel* d_velocityDivergenceBCLabel;
      const VarLabel* d_continuityResidualLabel;

      const VarLabel* d_InitNormLabel;
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
    }; // End class ArchesLabel
} // End namespace Uintah


#endif

