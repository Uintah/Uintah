//----- ArchesLabel.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

using namespace Uintah;

//****************************************************************************
// Default constructor for ArchesLabel
//****************************************************************************
ArchesLabel::ArchesLabel()
{
  int noStencil = 7;
  d_stencilMatl = scinew MaterialSubset();
  for (int i = 0; i < noStencil; i++)
    d_stencilMatl->add(i);
  d_stencilMatl->addReference();

  // Cell Information
  d_cellInfoLabel = scinew VarLabel("cellInformation",
			    PerPatch<CellInformationP>::getTypeDescription());
  // Cell type
  d_cellTypeLabel = scinew VarLabel("cellType", 
				  CCVariable<int>::getTypeDescription() );
  // labels for inlet and outlet flow rates
  d_totalflowINLabel = scinew VarLabel("totalflowIN",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowOUTLabel = scinew VarLabel("totalflowOUT",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowOUToutbcLabel = scinew VarLabel("totalflowOUToutbc",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalAreaOUTLabel = scinew VarLabel("totalAreaOUT",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_denAccumLabel = scinew VarLabel("denAccum",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 

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

  // reactive scalar

  d_reactscalarINLabel = scinew VarLabel("reactscalarIN",
				    CCVariable<double>::getTypeDescription() );
  d_reactscalarSPLabel = scinew VarLabel("reactscalarSP",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarOUTBCLabel = scinew VarLabel("reactscalarOUTBC",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarCPBCLabel = scinew VarLabel("reactscalarCPBC",
				      CCVariable<double>::getTypeDescription() );

  // reactscalar variance
  d_reactscalarVarINLabel = scinew VarLabel("reactscalarVarIN", 
				       CCVariable<double>::getTypeDescription() );

  d_reactscalarVarSPLabel = scinew VarLabel("reactscalarVarSP", 
				       CCVariable<double>::getTypeDescription() );
  // Reactscalar Coef
  d_reactscalCoefSBLMLabel = scinew VarLabel("reactscalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Conv Coef
  d_reactscalConvCoefSBLMLabel = scinew VarLabel("reactscalConvCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Linear Src
  d_reactscalLinSrcSBLMLabel = scinew VarLabel("reactscalLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Non Linear Src
  d_reactscalNonLinSrcSBLMLabel = scinew VarLabel("reactscalNonLinSrcSBLM",
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

  d_reactscalarRes = scinew VarLabel("reactscalarRes",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyRes = scinew VarLabel("enthalpyRes",
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

  // for pressure grad term in momentum

  d_pressGradUSuLabel = scinew VarLabel("pressGradUSu",
					SFCXVariable<double>::getTypeDescription() );
  d_pressGradVSuLabel = scinew VarLabel("pressGradVSu",
					SFCYVariable<double>::getTypeDescription() );
  d_pressGradWSuLabel = scinew VarLabel("pressGradWSu",
					SFCZVariable<double>::getTypeDescription() );
  // multimaterial labels

  // multimaterial wall/intrusion cells

  d_mmcellTypeLabel = scinew VarLabel("mmcellType",
				      CCVariable<int>::getTypeDescription() );

  // Label for void fraction, after correction for wall cells using cutoff

  d_mmgasVolFracLabel = scinew VarLabel("mmgasVolFrac",
					CCVariable<double>::getTypeDescription() );

    // for reacting flows
  d_tempINLabel = scinew VarLabel("tempIN",
				  CCVariable<double>::getTypeDescription() );
  d_co2INLabel = scinew VarLabel("co2IN",
				  CCVariable<double>::getTypeDescription() );


  // Array containing the reference density multiplied by the void fraction
  // used for correct reference density subtraction in the multimaterial
  // case

  d_denRefArrayLabel = scinew VarLabel("denRefArray",
					CCVariable<double>::getTypeDescription() );

  // Microscopic density (i.e., without void fraction) of gas

  d_densityMicroLabel = scinew VarLabel("denMicro",
					CCVariable<double>::getTypeDescription() );
  d_densityMicroINLabel = scinew VarLabel("denMicroIN",
					CCVariable<double>::getTypeDescription() );

  // Label for the sum of the relative pressure and the hydrostatic 
  // contribution

  d_pressPlusHydroLabel = scinew VarLabel("pPlusHydro",
					CCVariable<double>::getTypeDescription() );

  d_scalarOUTBCLabel =  scinew VarLabel("scalarOUTBC",
					CCVariable<double>::getTypeDescription() );
  d_uVelocityOUTBCLabel = scinew VarLabel("uVelocityOUTBC",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelocityOUTBCLabel= scinew VarLabel("vVelocityOUTBC",
				  SFCYVariable<double>::getTypeDescription() );
  d_wVelocityOUTBCLabel= scinew VarLabel("wVelocityOUTBC",
				   SFCZVariable<double>::getTypeDescription() );

  d_uvwoutLabel = scinew VarLabel("uvwout",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 


  // predictor-corrector labels
  // Scalar Coef
  d_scalCoefPredLabel = scinew VarLabel("scalCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Conv Coef
  d_scalConvCoefPredLabel = scinew VarLabel("scalConvCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Linear Src
  d_scalLinSrcPredLabel = scinew VarLabel("scalLinSrcPred",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Non Linear Src
  d_scalNonLinSrcPredLabel = scinew VarLabel("scalNonLinSrcPred",
				   CCVariable<double>::getTypeDescription() );

  d_scalCoefCorrLabel = scinew VarLabel("scalCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Conv Coef
  d_scalConvCoefCorrLabel = scinew VarLabel("scalConvCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Linear Src
  d_scalLinSrcCorrLabel = scinew VarLabel("scalLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Non Linear Src
  d_scalNonLinSrcCorrLabel = scinew VarLabel("scalNonLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );

  d_scalarPredLabel = scinew VarLabel("scalarPred",
				   CCVariable<double>::getTypeDescription() );


  // predictor-corrector labels
  // Reactscalar Coef
  d_reactscalCoefPredLabel = scinew VarLabel("reactscalCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Conv Coef
  d_reactscalConvCoefPredLabel = scinew VarLabel("reactscalConvCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Linear Src
  d_reactscalLinSrcPredLabel = scinew VarLabel("reactscalLinSrcPred",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Non Linear Src
  d_reactscalNonLinSrcPredLabel = scinew VarLabel("reactscalNonLinSrcPred",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalCoefCorrLabel = scinew VarLabel("reactscalCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Conv Coef
  d_reactscalConvCoefCorrLabel = scinew VarLabel("reactscalConvCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Linear Src
  d_reactscalLinSrcCorrLabel = scinew VarLabel("reactscalLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Non Linear Src
  d_reactscalNonLinSrcCorrLabel = scinew VarLabel("reactscalNonLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarPredLabel = scinew VarLabel("reactscalarPred",
				   CCVariable<double>::getTypeDescription() );

  d_densityPredLabel = scinew VarLabel("densityPred",
				   CCVariable<double>::getTypeDescription() );

  d_uVelRhoHatLabel = scinew VarLabel("uvelRhoHat",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelRhoHatLabel= scinew VarLabel("vvelRhoHat",
				  SFCYVariable<double>::getTypeDescription() );
  d_wVelRhoHatLabel= scinew VarLabel("wvelRhoHat",
				   SFCZVariable<double>::getTypeDescription() );

  d_pressurePredLabel = scinew VarLabel("pressurePred", 
				    CCVariable<double>::getTypeDescription() );
  // Pressure Coeff Labels
  d_presCoefCorrLabel = scinew VarLabel("presCoefCorr", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Linear Src Labels
  d_presLinSrcCorrLabel = scinew VarLabel("presLinSrcCorr", 
					CCVariable<double>::getTypeDescription() );
  // Pressure Non Linear Src Labels
  d_presNonLinSrcCorrLabel = scinew VarLabel("presNonLinSrcCorr", 
					   CCVariable<double>::getTypeDescription() );
  // U-Velocity Labels
  d_uVelocityPredLabel = scinew VarLabel("uVelocityPred", 
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityPredLabel = scinew VarLabel("vVelocityPred", 
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityPredLabel = scinew VarLabel("wVelocityPred", 
				    SFCZVariable<double>::getTypeDescription() );

  // enthalpy labels
  // Enthalpy 
  d_enthalpyINLabel = scinew VarLabel("enthalpyIN",
				    CCVariable<double>::getTypeDescription() );
  d_enthalpySPLabel = scinew VarLabel("enthalpySP",
				   CCVariable<double>::getTypeDescription() );
  d_enthalpySPBCLabel = scinew VarLabel("enthalpySPBC",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyRXNLabel = scinew VarLabel("enthalpyRXN",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyCPBCLabel = scinew VarLabel("enthalpyCPBC",
				      CCVariable<double>::getTypeDescription() );
  d_enthalpyOUTBCLabel = scinew VarLabel("enthalpyOUTBC",
				      CCVariable<double>::getTypeDescription() );

  // Enthalpy Coef
  d_enthCoefSBLMLabel = scinew VarLabel("enthCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Conv Coef
  d_enthConvCoefSBLMLabel = scinew VarLabel("enthConvCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Linear Src
  d_enthLinSrcSBLMLabel = scinew VarLabel("enthLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Non Linear Src
  d_enthNonLinSrcSBLMLabel = scinew VarLabel("enthNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );

  // predictor-corrector labels
  // Enthalpy Coef
  d_enthCoefPredLabel = scinew VarLabel("enthCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Conv Coef
  d_enthConvCoefPredLabel = scinew VarLabel("enthConvCoefPred",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Linear Src
  d_enthLinSrcPredLabel = scinew VarLabel("enthLinSrcPred",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Non Linear Src
  d_enthNonLinSrcPredLabel = scinew VarLabel("enthNonLinSrcPred",
				   CCVariable<double>::getTypeDescription() );

  d_enthCoefCorrLabel = scinew VarLabel("enthCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Conv Coef
  d_enthConvCoefCorrLabel = scinew VarLabel("enthConvCoefCorr",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Linear Src
  d_enthLinSrcCorrLabel = scinew VarLabel("enthLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Non Linear Src
  d_enthNonLinSrcCorrLabel = scinew VarLabel("enthNonLinSrcCorr",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyPredLabel = scinew VarLabel("enthalpyPred",
				   CCVariable<double>::getTypeDescription() );
  // for radiation
  d_absorpINLabel = scinew VarLabel("absorpIN",
				    CCVariable<double>::getTypeDescription() );
  d_sootFVINLabel = scinew VarLabel("sootFVIN",
				    CCVariable<double>::getTypeDescription() );

  d_reactscalarSRCINLabel = scinew VarLabel("reactscalarSRCIN",
				    CCVariable<double>::getTypeDescription() );


}

//****************************************************************************
// Destructor
//****************************************************************************
ArchesLabel::~ArchesLabel()
{
  if (d_stencilMatl->removeReference())
    delete d_stencilMatl;
}

void ArchesLabel::setSharedState(SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
}
