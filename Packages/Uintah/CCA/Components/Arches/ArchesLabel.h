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
      const VarLabel* d_totalflowOUToutbcLabel;
      // net outlet area, mass balance, and overall outlet velocity
      const VarLabel* d_totalAreaOUTLabel;
      const VarLabel* d_denAccumLabel;
      // Density Labels
      const VarLabel* d_enthalpyRes;

      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_densityINLabel;

      // for old_dw in computeProps
      const VarLabel* d_densityCPLabel;

      // for computing divergence constraint
      const VarLabel* d_drhodfCPLabel;
      const VarLabel* d_drhodfPredLabel;

      // computed for old_dw in setProfile
      const VarLabel* d_densitySPLabel;

      // Viscosity Labels
      // computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_viscosityINLabel; 

      // for old_dw in computeTurbModel
      const VarLabel* d_viscosityCTSLabel;

      const VarLabel* d_viscosityPredLabel;


      // Pressure Labels

      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_pressureINLabel;  

      // for old_dw in computePressureBC
      const VarLabel* d_pressureSPBCLabel;

      // for new_dw in linearSolver
      const VarLabel* d_pressurePSLabel;

      // Pressure Coeff Labels
      // for new_dw in pressuresolver::linearizeMatrix
      const VarLabel* d_presCoefPBLMLabel;

      // Pressure Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presLinSrcPBLMLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // U-Velocity Labels

      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_uVelocityINLabel; 

      // computed for old_dw in setProfile
      const VarLabel* d_uVelocitySPLabel;

      // for old_dw in computePressureBC and for new_dw in linearsolve
      const VarLabel* d_uVelocitySPBCLabel;

      // for new_dw in inletvelocitybc
      const VarLabel* d_uVelocitySIVBCLabel;
      const VarLabel* d_uVelocityCPBCLabel;

      // U-Velocity Coeff Labels
      // matrix_dw in pressuresolver and momentum solver
      const VarLabel* d_uVelCoefPBLMLabel;

      // U-Velocity Convection Coeff Labels computed in pressuresolver and momentumsolver
      const VarLabel* d_uVelConvCoefPBLMLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* d_uVelLinSrcPBLMLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* d_uVelNonLinSrcPBLMLabel;

      // matrix_dw in pressuresolver and momentum solver
      const VarLabel* d_uVelCoefMBLMLabel;

      // U-Velocity Convection Coeff Labels computed in pressuresolver and momentumsolver
      const VarLabel* d_uVelConvCoefMBLMLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* d_uVelLinSrcMBLMLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* d_uVelNonLinSrcMBLMLabel;

      // V-Velocity Labels
      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_vVelocityINLabel; 

      // computed for old_dw in setProfile
      const VarLabel* d_vVelocitySPLabel;

      // for old_dw in computePressureBC
      const VarLabel* d_vVelocitySPBCLabel;

      // for new_dw in inletvelocitybc
      const VarLabel* d_vVelocitySIVBCLabel;
      const VarLabel* d_vVelocityCPBCLabel;

      // V-Velocity Coeff Labels
      const VarLabel* d_vVelCoefMBLMLabel;

      // V-Velocity Convection Coeff Labels
      const VarLabel* d_vVelConvCoefMBLMLabel;

      // V-Velocity Linear Src Label
      const VarLabel* d_vVelLinSrcMBLMLabel;

      // V-Velocity Non Linear Src Label
      const VarLabel* d_vVelNonLinSrcMBLMLabel;

      // V-Velocity Coeff Labels
      const VarLabel* d_vVelCoefPBLMLabel;

      // V-Velocity Convection Coeff Labels
      const VarLabel* d_vVelConvCoefPBLMLabel;

      // V-Velocity Linear Src Label
      const VarLabel* d_vVelLinSrcPBLMLabel;

      // V-Velocity Non Linear Src Label
      const VarLabel* d_vVelNonLinSrcPBLMLabel;

      // W-Velocity Labels

      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_wVelocityINLabel; 

      // computed for old_dw in setProfile
      const VarLabel* d_wVelocitySPLabel;

      // for old_dw in computePressureBC
      const VarLabel* d_wVelocitySPBCLabel;

      // for new_dw in inletvelocitybc
      const VarLabel* d_wVelocitySIVBCLabel;
      const VarLabel* d_wVelocityCPBCLabel;

      // W-Velocity Coeff Labels
      const VarLabel* d_wVelCoefPBLMLabel;

      // W-Velocity Convection Coeff Labels
      const VarLabel* d_wVelConvCoefPBLMLabel;

      // W-Velocity Linear Src Label
      const VarLabel* d_wVelLinSrcPBLMLabel;

      // W-Velocity Non Linear Src Label
      const VarLabel* d_wVelNonLinSrcPBLMLabel;

      // W-Velocity Coeff Labels
      const VarLabel* d_wVelCoefMBLMLabel;

      // W-Velocity Convection Coeff Labels
      const VarLabel* d_wVelConvCoefMBLMLabel;

      // W-Velocity Linear Src Label
      const VarLabel* d_wVelLinSrcMBLMLabel;

      // W-Velocity Non Linear Src Label
      const VarLabel* d_wVelNonLinSrcMBLMLabel;

      // Scalar Labels
      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess

      const VarLabel* d_scalarINLabel;  

      // computed for old_dw in setProfile

      const VarLabel* d_scalarSPLabel;

      // computed as a part of pressure boundary calculations

      const VarLabel* d_scalarCPBCLabel;

      // Scalar variance labels
      // computed for old_dw in paramInit and for
      // new_dw in setInitailGuess

      const VarLabel* d_scalarVarINLabel;

      // computed for new_dw in Smagorinsky Model

      const VarLabel* d_scalarVarSPLabel;
      // required for flamelet model
      const VarLabel* d_scalarDissINLabel;

      // computed for new_dw in Smagorinsky Model fir flamelet

      const VarLabel* d_scalarDissSPLabel;

      // Scalar Coef

      const VarLabel* d_scalCoefSBLMLabel;

      // scalar diffusion coeffs, required for divergence constraint
      const VarLabel* d_scalDiffCoefPredLabel;

      const VarLabel* d_scalDiffCoefSrcPredLabel;

      const VarLabel* d_scalDiffCoefCorrLabel;

      const VarLabel* d_enthDiffCoefPredLabel;

      const VarLabel* d_enthDiffCoefCorrLabel;

      const VarLabel* d_reactscalDiffCoefPredLabel;

      const VarLabel* d_reactscalDiffCoefCorrLabel;

      // Scalar Conv Coef

      const VarLabel* d_scalConvCoefSBLMLabel;

      // Scalar Linear Src

      const VarLabel* d_scalLinSrcSBLMLabel;

      // Scalar NonLinear Src

      const VarLabel* d_scalNonLinSrcSBLMLabel;



      // reactive scalars


      const VarLabel* d_reactscalarINLabel;  

      // computed for old_dw in setProfile

      const VarLabel* d_reactscalarSPLabel;

      // computed as a part of pressure boundary calculations

      const VarLabel* d_reactscalarCPBCLabel;

      // Reactscalar variance labels
      // computed for old_dw in paramInit and for
      // new_dw in setInitailGuess

      const VarLabel* d_reactscalarVarINLabel;

      // computed for new_dw in Smagorinsky Model

      const VarLabel* d_reactscalarVarSPLabel;

      // Reactscalar Coef

      const VarLabel* d_reactscalCoefSBLMLabel;

      // Reactscalar Conv Coef

      const VarLabel* d_reactscalConvCoefSBLMLabel;

      // Reactscalar Linear Src

      const VarLabel* d_reactscalLinSrcSBLMLabel;

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

      // labels required for explicit solver to store old guess values

      const VarLabel* d_old_uVelocityGuess;
      const VarLabel* d_old_vVelocityGuess;
      const VarLabel* d_old_wVelocityGuess;
      const VarLabel* d_old_scalarGuess;

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
      const VarLabel* d_co2INLabel;
      const VarLabel* d_h2oINLabel;
      const VarLabel* d_denRefArrayLabel;
      const VarLabel* d_denRefArrayPredLabel;
      const VarLabel* d_densityMicroLabel;
      const VarLabel* d_densityMicroINLabel;
      const VarLabel* d_pressPlusHydroLabel;
      // predicted
      const VarLabel* d_tempINPredLabel;
      const VarLabel* d_co2INPredLabel;
      const VarLabel* d_h2oINPredLabel;

      // for outlet bc
      const VarLabel* d_uvwoutLabel;
      const VarLabel* d_uVelocityOUTBCLabel;
      const VarLabel* d_vVelocityOUTBCLabel;
      const VarLabel* d_wVelocityOUTBCLabel;
      const VarLabel* d_scalarOUTBCLabel;
      // pred-corr labels
      const VarLabel* d_scalCoefPredLabel;

      // Scalar Conv Coef

      const VarLabel* d_scalConvCoefPredLabel;

      // Scalar Linear Src

      const VarLabel* d_scalLinSrcPredLabel;

      // Scalar NonLinear Src

      const VarLabel* d_scalNonLinSrcPredLabel;
      // scalar pred
      const VarLabel* d_scalarPredLabel;

      const VarLabel* d_scalCoefCorrLabel;

      // Scalar Conv Coef

      const VarLabel* d_scalConvCoefCorrLabel;

      // Scalar Linear Src

      const VarLabel* d_scalLinSrcCorrLabel;

      // Scalar NonLinear Src

      const VarLabel* d_scalNonLinSrcCorrLabel;



      // for reactive scalar
      const VarLabel* d_reactscalarOUTBCLabel;
      // pred-corr labels
      const VarLabel* d_reactscalCoefPredLabel;

      // Scalar Conv Coef

      const VarLabel* d_reactscalConvCoefPredLabel;

      // Scalar Linear Src

      const VarLabel* d_reactscalLinSrcPredLabel;

      // Scalar NonLinear Src

      const VarLabel* d_reactscalNonLinSrcPredLabel;
      // scalar pred
      const VarLabel* d_reactscalarPredLabel;

      const VarLabel* d_reactscalCoefCorrLabel;

      // Scalar Conv Coef

      const VarLabel* d_reactscalConvCoefCorrLabel;

      // Scalar Linear Src

      const VarLabel* d_reactscalLinSrcCorrLabel;

      // Scalar NonLinear Src

      const VarLabel* d_reactscalNonLinSrcCorrLabel;
      // for corrector
      const VarLabel* d_uVelCoefPBLMCorrLabel;

      // U-Velocity Convection Coeff Labels computed in pressuresolver and momentumsolver
      const VarLabel* d_uVelConvCoefPBLMCorrLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* d_uVelLinSrcPBLMCorrLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* d_uVelNonLinSrcPBLMCorrLabel;

      const VarLabel* d_vVelCoefPBLMCorrLabel;

      // U-Velocity Convection Coeff Labels computed in pressuresolver and momentumsolver
      const VarLabel* d_vVelConvCoefPBLMCorrLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* d_vVelLinSrcPBLMCorrLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* d_vVelNonLinSrcPBLMCorrLabel;


      const VarLabel* d_wVelCoefPBLMCorrLabel;

      // U-Velocity Convection Coeff Labels computed in pressuresolver and momentumsolver
      const VarLabel* d_wVelConvCoefPBLMCorrLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* d_wVelLinSrcPBLMCorrLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* d_wVelNonLinSrcPBLMCorrLabel;

      const VarLabel* d_uVelRhoHatCorrLabel;
      const VarLabel* d_vVelRhoHatCorrLabel;
      const VarLabel* d_wVelRhoHatCorrLabel;


      // for density
      const VarLabel* d_densityPredLabel;
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
      const VarLabel* d_pressureCorrSPBCLabel;

      const VarLabel* d_presCoefCorrLabel;

      // Pressure Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presLinSrcCorrLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcCorrLabel;
      
      const VarLabel* d_uVelocityPredLabel;
      const VarLabel* d_vVelocityPredLabel;
      const VarLabel* d_wVelocityPredLabel;
      // for enthalpy equation
      const VarLabel* d_enthalpyINLabel;  

      // computed for old_dw in setProfile
      const VarLabel* d_enthalpySPBCLabel;
      const VarLabel* d_enthalpySPLabel;
      // for validation
      const VarLabel* d_enthalpyRXNLabel;
      const VarLabel* d_enthalpyRXNPredLabel;
      // computed as a part of pressure boundary calculations

      const VarLabel* d_enthalpyCPBCLabel;
      const VarLabel* d_enthalpyOUTBCLabel;

      // Enthalpy Coef

      const VarLabel* d_enthCoefSBLMLabel;

      // Enthalpy Conv Coef

      const VarLabel* d_enthConvCoefSBLMLabel;

      // Enthalpy Linear Src

      const VarLabel* d_enthLinSrcSBLMLabel;

      // Enthalpy NonLinear Src

      const VarLabel* d_enthNonLinSrcSBLMLabel;

      // pred-corr labels for enthalpy
      const VarLabel* d_enthCoefPredLabel;

      // Scalar Conv Coef

      const VarLabel* d_enthConvCoefPredLabel;

      // Scalar Linear Src

      const VarLabel* d_enthLinSrcPredLabel;

      // Scalar NonLinear Src

      const VarLabel* d_enthNonLinSrcPredLabel;
      // scalar pred
      const VarLabel* d_enthalpyPredLabel;

      const VarLabel* d_enthCoefCorrLabel;

      // Scalar Conv Coef

      const VarLabel* d_enthConvCoefCorrLabel;

      // Scalar Linear Src

      const VarLabel* d_enthLinSrcCorrLabel;

      // Scalar NonLinear Src

      const VarLabel* d_enthNonLinSrcCorrLabel;
      // for radiation
      const VarLabel* d_absorpINLabel;
      const VarLabel* d_sootFVINLabel;
      // reactive scalar source term from properties
      const VarLabel* d_reactscalarSRCINLabel;
      const VarLabel* d_absorpINPredLabel;
      const VarLabel* d_sootFVINPredLabel;
      // reactive scalar source term from properties
      const VarLabel* d_reactscalarSRCINPredLabel;
      
      // runge-kutta 3d order scalar labels
      const VarLabel* d_scalCoefIntermLabel;
      const VarLabel* d_scalNonLinSrcIntermLabel;
      const VarLabel* d_scalarIntermLabel;
      const VarLabel* d_scalConvCoefIntermLabel;
      const VarLabel* d_scalDiffCoefIntermLabel;
      const VarLabel* d_scalLinSrcIntermLabel;
      const VarLabel* d_scalarTempLabel;
      
      // runge-kutta 3d order enthalpy labels
      const VarLabel* d_enthCoefIntermLabel;
      const VarLabel* d_enthNonLinSrcIntermLabel;
      const VarLabel* d_enthalpyIntermLabel;
      const VarLabel* d_enthConvCoefIntermLabel;
      const VarLabel* d_enthDiffCoefIntermLabel;
      const VarLabel* d_enthLinSrcIntermLabel;
      const VarLabel* d_enthalpyTempLabel;

      // runge-kutta 3d order reactscalar labels
      const VarLabel* d_reactscalCoefIntermLabel;
      const VarLabel* d_reactscalNonLinSrcIntermLabel;
      const VarLabel* d_reactscalarIntermLabel;
      const VarLabel* d_reactscalConvCoefIntermLabel;
      const VarLabel* d_reactscalDiffCoefIntermLabel;
      const VarLabel* d_reactscalLinSrcIntermLabel;
      const VarLabel* d_reactscalarTempLabel;
      
      // runge-kutta 3d order properties labels
      const VarLabel* d_densityIntermLabel;     
      const VarLabel* d_viscosityIntermLabel;
      const VarLabel* d_drhodfIntermLabel;
      const VarLabel* d_tempINIntermLabel;
      const VarLabel* d_co2INIntermLabel; 
      const VarLabel* d_h2oINIntermLabel; 
      const VarLabel* d_enthalpyRXNIntermLabel; 
      const VarLabel* d_reactscalarSRCINIntermLabel; 
      const VarLabel* d_absorpINIntermLabel; 
      const VarLabel* d_sootFVINIntermLabel; 
      const VarLabel* d_denRefArrayIntermLabel;
      const VarLabel* d_refDensityInterm_label;
      
      // runge-kutta 3d order pressure and momentum labels
      const VarLabel* d_uVelCoefPBLMIntermLabel;
      const VarLabel* d_uVelConvCoefPBLMIntermLabel;
      const VarLabel* d_uVelLinSrcPBLMIntermLabel;
      const VarLabel* d_uVelNonLinSrcPBLMIntermLabel;
      const VarLabel* d_vVelCoefPBLMIntermLabel;
      const VarLabel* d_vVelConvCoefPBLMIntermLabel;
      const VarLabel* d_vVelLinSrcPBLMIntermLabel;
      const VarLabel* d_vVelNonLinSrcPBLMIntermLabel;
      const VarLabel* d_wVelCoefPBLMIntermLabel;
      const VarLabel* d_wVelConvCoefPBLMIntermLabel;
      const VarLabel* d_wVelLinSrcPBLMIntermLabel;
      const VarLabel* d_wVelNonLinSrcPBLMIntermLabel;
      const VarLabel* d_uVelRhoHatIntermLabel;
      const VarLabel* d_vVelRhoHatIntermLabel;
      const VarLabel* d_wVelRhoHatIntermLabel;
      const VarLabel* d_pressureIntermLabel;
      const VarLabel* d_presCoefIntermLabel;
      const VarLabel* d_presLinSrcIntermLabel;
      const VarLabel* d_presNonLinSrcIntermLabel;
      const VarLabel* d_uVelTempLabel;
      const VarLabel* d_vVelTempLabel;
      const VarLabel* d_wVelTempLabel;
      const VarLabel* d_uVelocityIntermLabel;
      const VarLabel* d_vVelocityIntermLabel;
      const VarLabel* d_wVelocityIntermLabel;
//      const VarLabel* d_velocityDivergenceLabel;
//      const VarLabel* d_velocityDivergenceBCLabel;

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
// labels for interpolated velocity used in Runge-Kutta method
      const VarLabel* d_newCCUVelocityPredLabel;
      const VarLabel* d_newCCVVelocityPredLabel;
      const VarLabel* d_newCCWVelocityPredLabel;
      const VarLabel* d_newCCUVelocityIntermLabel;
      const VarLabel* d_newCCVVelocityIntermLabel;
      const VarLabel* d_newCCWVelocityIntermLabel;
// filtered convection terms in momentum eqn
      const VarLabel* d_filteredRhoUjULabel;
      const VarLabel* d_filteredRhoUjVLabel;
      const VarLabel* d_filteredRhoUjWLabel;
// kinetic energy
      const VarLabel* d_kineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyPredLabel;
      const VarLabel* d_totalKineticEnergyIntermLabel;
 
    }; // End class ArchesLabel
} // End namespace Uintah


#endif

