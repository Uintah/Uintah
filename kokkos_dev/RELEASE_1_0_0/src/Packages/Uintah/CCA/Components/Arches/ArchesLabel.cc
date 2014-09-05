//----- ArchesLabel.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

//****************************************************************************
// Default constructor for ArchesLabel
//****************************************************************************
ArchesLabel::ArchesLabel()
{


  // Cell Information
  d_cellInfoLabel = scinew VarLabel("cellInformation",
			    PerPatch<CellInformationP>::getTypeDescription());
  // Cell type
  d_cellTypeLabel = scinew VarLabel("cellType", 
				  CCVariable<int>::getTypeDescription() );
  // Density Labels
  d_densityINLabel = scinew VarLabel("densityIN", 
				   CCVariable<double>::getTypeDescription() );
  d_densitySPLabel = scinew VarLabel("densitySP", 
				   CCVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP", 
				  CCVariable<double>::getTypeDescription() );
  // Viscosity Labels
  d_viscosityINLabel = scinew VarLabel("viscosityIN", 
				   CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Labels
  d_pressureINLabel = scinew VarLabel("pressureIN", 
				    CCVariable<double>::getTypeDescription() );
  d_pressureSPBCLabel = scinew VarLabel("pressureSPBC", 
				      CCVariable<double>::getTypeDescription() );
  d_pressurePSLabel = scinew VarLabel("pressurePS", 
				    CCVariable<double>::getTypeDescription() );
  // Pressure Coeff Labels
  d_presCoefPBLMLabel = scinew VarLabel("presCoefPBLM", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Linear Src Labels
  d_presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM", 
					CCVariable<double>::getTypeDescription() );
  // Pressure Non Linear Src Labels
  d_presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM", 
					   CCVariable<double>::getTypeDescription() );
  // U-Velocity Labels
  d_uVelocityINLabel = scinew VarLabel("uVelocityIN", 
				    SFCXVariable<double>::getTypeDescription() );
  d_uVelocitySPLabel = scinew VarLabel("uVelocitySP", 
				    SFCXVariable<double>::getTypeDescription() );
  d_uVelocitySPBCLabel = scinew VarLabel("uVelocitySPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC", 
				       SFCXVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Coeff Labels
  d_uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
			       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Convection Coeff Labels
  d_uVelConvCoefPBLMLabel = scinew VarLabel("uVelConvCoefPBLM",
				        SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels
  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				 SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Coeff Labels
  d_uVelCoefMBLMLabel = scinew VarLabel("uVelCoefMBLM",
			       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Convection Coeff Labels
  d_uVelConvCoefMBLMLabel = scinew VarLabel("uVelConvCoefMBLM",
				        SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				 SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  // V-Velocity Labels
  d_vVelocityINLabel = scinew VarLabel("vVelocityIN", 
				    SFCYVariable<double>::getTypeDescription() );
  d_vVelocitySPLabel = scinew VarLabel("vVelocitySP", 
				    SFCYVariable<double>::getTypeDescription() );
  d_vVelocitySPBCLabel = scinew VarLabel("vVelocitySPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC", 
				       SFCYVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Coeff Labels
  d_vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
			       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Convection Coeff Labels
  d_vVelConvCoefPBLMLabel = scinew VarLabel("vVelConvCoefPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				 SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Coeff Labels
  d_vVelCoefMBLMLabel = scinew VarLabel("vVelCoefMBLM",
			       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Convection Coeff Labels
  d_vVelConvCoefMBLMLabel = scinew VarLabel("vVelConvCoefMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				 SFCYVariable<double>::getTypeDescription() );

  // labels for ref density and pressure
  d_refDensity_label = scinew VarLabel("refDensityLabel",
				       sum_vartype::getTypeDescription() );
  d_refPressure_label = scinew VarLabel("refPressureLabel",
				       sum_vartype::getTypeDescription() );

  // V-Velocity Non Linear Src Labels
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  // W-Velocity Labels
  d_wVelocityINLabel = scinew VarLabel("wVelocityIN", 
				    SFCZVariable<double>::getTypeDescription() );
  d_wVelocitySPLabel = scinew VarLabel("wVelocitySP", 
				    SFCZVariable<double>::getTypeDescription() );
  d_wVelocitySPBCLabel = scinew VarLabel("wVelocitySPBC", 
				       SFCZVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC", 
				       SFCZVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC", 
				       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Coeff Labels
  d_wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
			       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Convection Coeff Labels
  d_wVelConvCoefPBLMLabel = scinew VarLabel("wVelConvCoefPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				 SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Coeff Labels
  d_wVelCoefMBLMLabel = scinew VarLabel("wVelCoefMBLM",
			       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Convection Coeff Labels
  d_wVelConvCoefMBLMLabel = scinew VarLabel("wVelConvCoefMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				 SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );
  // Scalar 
  d_scalarINLabel = scinew VarLabel("scalarIN",
				    CCVariable<double>::getTypeDescription() );
  d_scalarSPLabel = scinew VarLabel("scalarSP",
				   CCVariable<double>::getTypeDescription() );

  d_scalarCPBCLabel = scinew VarLabel("scalarCPBC",
				      CCVariable<double>::getTypeDescription() );

  // scalar variance
  d_scalarVarINLabel = scinew VarLabel("scalarVarIN", 
				       CCVariable<double>::getTypeDescription() );

  d_scalarVarSPLabel = scinew VarLabel("scalarVarSP", 
				       CCVariable<double>::getTypeDescription() );
  // Scalar Coef
  d_scalCoefSBLMLabel = scinew VarLabel("scalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Conv Coef
  d_scalConvCoefSBLMLabel = scinew VarLabel("scalConvCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Linear Src
  d_scalLinSrcSBLMLabel = scinew VarLabel("scalLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Non Linear Src
  d_scalNonLinSrcSBLMLabel = scinew VarLabel("scalNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // labels for nonlinear residuals
  d_presResidPSLabel = scinew VarLabel("presResidPSLabel",
				        ReductionVariable<double,
				       Reductions::Sum<double> >::getTypeDescription());
  d_presTruncPSLabel = scinew VarLabel("presTruncPSLabel",
				       sum_vartype::getTypeDescription() );
  d_uVelResidPSLabel = scinew VarLabel("uVelResidPSLabel",
				       sum_vartype::getTypeDescription() );
  d_uVelTruncPSLabel = scinew VarLabel("uVelTruncPSLabel",
				       sum_vartype::getTypeDescription() );
  d_vVelResidPSLabel = scinew VarLabel("vVelResidPSLabel",
				       sum_vartype::getTypeDescription() );
  d_vVelTruncPSLabel = scinew VarLabel("vVelTruncPSLabel",
				       sum_vartype::getTypeDescription() );
  d_wVelResidPSLabel = scinew VarLabel("wVelResidPSLabel",
				       sum_vartype::getTypeDescription() );
  d_wVelTruncPSLabel = scinew VarLabel("wVelTruncPSLabel",
				       sum_vartype::getTypeDescription() );
  d_scalarResidLabel = scinew VarLabel("scalarResidLabel",
				       sum_vartype::getTypeDescription() );
  d_scalarTruncLabel = scinew VarLabel("scalarTruncLabel",
				       sum_vartype::getTypeDescription() );


  d_pressureRes = scinew VarLabel("pressureRes",
				   CCVariable<double>::getTypeDescription() );
  d_uVelocityRes = scinew VarLabel("uVelocityRes",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelocityRes = scinew VarLabel("vVelocityRes",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelocityRes = scinew VarLabel("wVelocityRes",
				   SFCZVariable<double>::getTypeDescription() );
  d_scalarRes = scinew VarLabel("scalarRes",
				   CCVariable<double>::getTypeDescription() );

  d_old_uVelocityGuess = scinew VarLabel("olduVelocityguess",
				       SFCXVariable<double>::getTypeDescription());
  d_old_vVelocityGuess = scinew VarLabel("oldvVelocityguess",
				       SFCYVariable<double>::getTypeDescription());
  d_old_wVelocityGuess = scinew VarLabel("oldwVelocityguess",
				       SFCZVariable<double>::getTypeDescription());
  d_old_scalarGuess = scinew VarLabel("oldscalarguess",
				       CCVariable<double>::getTypeDescription());

  // Unsure stuff
  // Unsure stuff
  d_DUPBLMLabel = scinew VarLabel("DUPBLM",
				SFCXVariable<double>::getTypeDescription() );
  d_DVPBLMLabel = scinew VarLabel("DVPBLM",
				SFCYVariable<double>::getTypeDescription() );
  d_DWPBLMLabel = scinew VarLabel("DWPBLM",
				SFCZVariable<double>::getTypeDescription() );
  d_DUMBLMLabel = scinew VarLabel("DUMBLM",
				SFCXVariable<double>::getTypeDescription() );
  d_DVMBLMLabel = scinew VarLabel("DVMBLM",
				SFCYVariable<double>::getTypeDescription() );
  d_DWMBLMLabel = scinew VarLabel("DWMBLM",
				SFCZVariable<double>::getTypeDescription() );

  // Labels that access the velocity stored as a cell centered vector
  // after interpolation (for use in visualization)
  d_oldCCVelocityLabel = scinew VarLabel("oldCCVelocity",
				CCVariable<Vector>::getTypeDescription() );
  d_newCCVelocityLabel = scinew VarLabel("newCCVelocity",
				CCVariable<Vector>::getTypeDescription() );
  d_newCCUVelocityLabel = scinew VarLabel("newCCUVelocity",
				CCVariable<double>::getTypeDescription() );
  d_newCCVVelocityLabel = scinew VarLabel("newCCVVelocity",
				CCVariable<double>::getTypeDescription() );
  d_newCCWVelocityLabel = scinew VarLabel("newCCWVelocity",
				CCVariable<double>::getTypeDescription() );

  // multimaterial labels
  d_mmcellTypeLabel = scinew VarLabel("mmcellType",
				      CCVariable<int>::getTypeDescription() );
  d_mmgasVolFracLabel = scinew VarLabel("mmgasVolFrac",
					CCVariable<double>::getTypeDescription() );

}

//****************************************************************************
// Destructor
//****************************************************************************
ArchesLabel::~ArchesLabel()
{
}

