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

      // Cell Information
      // for old_dw, perpatch var

      const VarLabel* d_cellInfoLabel;

      // Cell type

      const VarLabel* d_cellTypeLabel;//computed for old_dw in cellTypeInit
     
      // Labels for inlet and flow rate
      const VarLabel* d_totalflowINLabel;
      const VarLabel* d_totalflowOUTLabel;
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

      // computed for old_dw in setProfile
      const VarLabel* d_densitySPLabel;

      // Viscosity Labels
      // computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess
      const VarLabel* d_viscosityINLabel; 

      // for old_dw in computeTurbModel
      const VarLabel* d_viscosityCTSLabel;

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

      // Scalar Coef

      const VarLabel* d_scalCoefSBLMLabel;

      // Scalar Conv Coef

      const VarLabel* d_scalConvCoefSBLMLabel;

      // Scalar Linear Src

      const VarLabel* d_scalLinSrcSBLMLabel;

      // Scalar NonLinear Src

      const VarLabel* d_scalNonLinSrcSBLMLabel;

      // label for ref_density and pressure

      const VarLabel* d_refDensity_label;
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
      const VarLabel* d_denRefArrayLabel;
      const VarLabel* d_densityMicroLabel;
      const VarLabel* d_densityMicroINLabel;
      const VarLabel* d_pressPlusHydroLabel;

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
      // for density
      const VarLabel* d_densityPredLabel;
      // labels for pressure solver
      const VarLabel* d_uVelRhoHatLabel;
      const VarLabel* d_vVelRhoHatLabel;
      const VarLabel* d_wVelRhoHatLabel;

      const VarLabel* d_pressurePredLabel;

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
    }; // End class ArchesLabel
} // End namespace Uintah


#endif

