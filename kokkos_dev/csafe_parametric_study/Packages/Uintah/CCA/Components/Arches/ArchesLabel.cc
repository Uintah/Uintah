//----- ArchesLabel.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

using namespace Uintah;

//****************************************************************************
// Default constructor for ArchesLabel
//****************************************************************************
ArchesLabel::ArchesLabel()
{
// Seven point stencil
  int numberStencilComponents = 7;
  d_stencilMatl = scinew MaterialSubset();
  for (int i = 0; i < numberStencilComponents; i++)
    d_stencilMatl->add(i);
  d_stencilMatl->addReference();

// Vector (1 order tensor)
  int numberVectorComponents = 3;
  d_vectorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberVectorComponents; i++)
    d_vectorMatl->add(i);
  d_vectorMatl->addReference();

// Second order tensor
  int numberTensorComponents = 9;
  d_tensorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberTensorComponents; i++)
    d_tensorMatl->add(i);
  d_tensorMatl->addReference();

// Second order symmetric tensor
  int numberSymTensorComponents = 6;
  d_symTensorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberSymTensorComponents; i++)
    d_symTensorMatl->add(i);
  d_symTensorMatl->addReference();

  // Cell Information
  d_cellInfoLabel = VarLabel::create("cellInformation",
			    PerPatch<CellInformationP>::getTypeDescription());
  // Cell type
  d_cellTypeLabel = VarLabel::create("cellType", 
				  CCVariable<int>::getTypeDescription() );
  // labels for inlet and outlet flow rates
  d_totalflowINLabel = VarLabel::create("totalflowIN",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowOUTLabel = VarLabel::create("totalflowOUT",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_netflowOUTBCLabel = VarLabel::create("netflowOUTBC",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalAreaOUTLabel = VarLabel::create("totalAreaOUT",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_denAccumLabel = VarLabel::create("denAccum",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 

  // Density Labels
  d_densityCPLabel = VarLabel::create("densityCP", 
				  CCVariable<double>::getTypeDescription() );
  d_densityEKTLabel = VarLabel::create("densityEKT", 
				  CCVariable<double>::getTypeDescription() );
  d_densityGuessLabel = VarLabel::create("densityGuess", 
				  CCVariable<double>::getTypeDescription() );
  d_densityTempLabel = VarLabel::create("densityTemp", 
				  CCVariable<double>::getTypeDescription() );
  d_densityOldOldLabel = VarLabel::create("densityOldOld", 
				  CCVariable<double>::getTypeDescription() );
  d_filterdrhodtLabel = VarLabel::create("filterdrhodt", 
				    CCVariable<double>::getTypeDescription() );
  d_drhodfCPLabel = VarLabel::create("drhodfCP", 
				  CCVariable<double>::getTypeDescription() );
  // Viscosity Labels
  d_viscosityCTSLabel = VarLabel::create("viscosityCTS", 
			      CCVariable<double>::getTypeDescription() );
  d_scalarDiffusivityLabel = VarLabel::create("scalarDiffusivity", 
			      CCVariable<double>::getTypeDescription() );
  d_enthalpyDiffusivityLabel = VarLabel::create("enthalpyDiffusivity", 
			      CCVariable<double>::getTypeDescription() );
  d_reactScalarDiffusivityLabel = VarLabel::create("reactScalarDiffusivity", 
			      CCVariable<double>::getTypeDescription() );
  // Pressure Labels
  d_pressurePSLabel = VarLabel::create("pressurePS", 
				      CCVariable<double>::getTypeDescription() );
  d_pressureExtraProjectionLabel = VarLabel::create("pressureExtraProjection", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Coeff Labels
  d_presCoefPBLMLabel = VarLabel::create("presCoefPBLM", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Non Linear Src Labels
  d_presNonLinSrcPBLMLabel = VarLabel::create("presNonLinSrcPBLM", 
					   CCVariable<double>::getTypeDescription() );
  // U-Velocity Labels
  d_uVelocitySPBCLabel = VarLabel::create("uVelocitySPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  // V-Velocity Labels
  d_vVelocitySPBCLabel = VarLabel::create("vVelocitySPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  // W-Velocity Labels
  d_wVelocitySPBCLabel = VarLabel::create("wVelocitySPBC", 
				       SFCZVariable<double>::getTypeDescription() );

  d_uVelocityEKTLabel = VarLabel::create("uVelocityEKT", 
				       SFCXVariable<double>::getTypeDescription() );
  d_vVelocityEKTLabel = VarLabel::create("vVelocityEKT", 
				       SFCYVariable<double>::getTypeDescription() );
  d_wVelocityEKTLabel = VarLabel::create("wVelocityEKT", 
				       SFCZVariable<double>::getTypeDescription() );
  // labels for ref density and pressure
  d_refDensity_label = VarLabel::create("refDensityLabel",
				       sum_vartype::getTypeDescription() );
  d_refDensityPred_label = VarLabel::create("refDensityPredLabel",
				       sum_vartype::getTypeDescription() );
  d_refPressure_label = VarLabel::create("refPressureLabel",
				       sum_vartype::getTypeDescription() );

  // Scalar 
  d_scalarSPLabel = VarLabel::create("scalarSP",
				   CCVariable<double>::getTypeDescription() );

  d_scalarEKTLabel = VarLabel::create("scalarEKT",
				   CCVariable<double>::getTypeDescription() );

  d_scalarTempLabel = VarLabel::create("scalarTemp",
				   CCVariable<double>::getTypeDescription() );

  d_scalarFELabel = VarLabel::create("scalarFE",
				   CCVariable<double>::getTypeDescription() );

  // scalar variance

  d_scalarVarSPLabel = VarLabel::create("scalarVarSP", 
				       CCVariable<double>::getTypeDescription() );

  d_scalarDissSPLabel = VarLabel::create("scalarDissSP", 
				       CCVariable<double>::getTypeDescription() );
  // Scalar Coef
  d_scalCoefSBLMLabel = VarLabel::create("scalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );


  // Scalar Non Linear Src
  d_scalNonLinSrcSBLMLabel = VarLabel::create("scalNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );

  // reactive scalar

  d_reactscalarSPLabel = VarLabel::create("reactscalarSP",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarEKTLabel = VarLabel::create("reactscalarEKT",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarTempLabel = VarLabel::create("reactscalarTemp",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalarFELabel = VarLabel::create("reactscalarFE",
				   CCVariable<double>::getTypeDescription() );

  // reactscalar variance
  d_reactscalarVarSPLabel = VarLabel::create("reactscalarVarSP", 
				       CCVariable<double>::getTypeDescription() );
  // Reactscalar Coef
  d_reactscalCoefSBLMLabel = VarLabel::create("reactscalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Reactscalar Non Linear Src
  d_reactscalNonLinSrcSBLMLabel = VarLabel::create("reactscalNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );


  // Labels that access the velocity stored as a cell centered vector
  // after interpolation (for use in visualization)

  d_oldCCVelocityLabel = VarLabel::create("oldCCVelocity",
				CCVariable<Vector>::getTypeDescription() );
  d_newCCVelocityLabel = VarLabel::create("newCCVelocity",
				CCVariable<Vector>::getTypeDescription() );
  d_newCCVelMagLabel = VarLabel::create("newCCVelMagnitude",
				CCVariable<double>::getTypeDescription() );
  d_newCCUVelocityLabel = VarLabel::create("newCCUVelocity",
				CCVariable<double>::getTypeDescription() );
  d_newCCVVelocityLabel = VarLabel::create("newCCVVelocity",
				CCVariable<double>::getTypeDescription() );
  d_newCCWVelocityLabel = VarLabel::create("newCCWVelocity",
				CCVariable<double>::getTypeDescription() );

  // multimaterial labels

  // multimaterial wall/intrusion cells

  d_mmcellTypeLabel = VarLabel::create("mmcellType",
				      CCVariable<int>::getTypeDescription() );

  // Label for void fraction, after correction for wall cells using cutoff

  d_mmgasVolFracLabel = VarLabel::create("mmgasVolFrac",
					CCVariable<double>::getTypeDescription() );

    // for reacting flows
  d_tempINLabel = VarLabel::create("tempIN",
				  CCVariable<double>::getTypeDescription() );
  d_cpINLabel = VarLabel::create("cpIN",
				  CCVariable<double>::getTypeDescription() );
  d_co2INLabel = VarLabel::create("co2IN",
				  CCVariable<double>::getTypeDescription() );
  d_normalizedScalarVarLabel = VarLabel::create("normalizedScalarVar",
				  CCVariable<double>::getTypeDescription() );
  d_heatLossLabel = VarLabel::create("heatLoss",
				  CCVariable<double>::getTypeDescription() );
  d_h2oINLabel = VarLabel::create("h2oIN",
				  CCVariable<double>::getTypeDescription() );
  d_h2sINLabel = VarLabel::create("h2sIN",
				  CCVariable<double>::getTypeDescription() );
  d_so2INLabel = VarLabel::create("so2IN",
				  CCVariable<double>::getTypeDescription() );
  d_so3INLabel = VarLabel::create("so3IN",
				  CCVariable<double>::getTypeDescription() );
  d_sulfurINLabel = VarLabel::create("sulfurIN",
				  CCVariable<double>::getTypeDescription() );
  d_coINLabel = VarLabel::create("coIN",
				  CCVariable<double>::getTypeDescription() );
  d_c2h2INLabel = VarLabel::create("c2h2IN",
                                    CCVariable<double>::getTypeDescription() );
  d_ch4INLabel = VarLabel::create("ch4IN",
                                   CCVariable<double>::getTypeDescription() );


  // Array containing the reference density multiplied by the void fraction
  // used for correct reference density subtraction in the multimaterial
  // case

  d_denRefArrayLabel = VarLabel::create("denRefArray",
					CCVariable<double>::getTypeDescription() );

  // Microscopic density (i.e., without void fraction) of gas

  d_densityMicroLabel = VarLabel::create("denMicro",
					CCVariable<double>::getTypeDescription() );
  d_densityMicroINLabel = VarLabel::create("denMicroIN",
					CCVariable<double>::getTypeDescription() );

  // Label for the sum of the relative pressure and the hydrostatic 
  // contribution

  d_pressPlusHydroLabel = VarLabel::create("pPlusHydro",
					CCVariable<double>::getTypeDescription() );


  d_uvwoutLabel = VarLabel::create("uvwout",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 


  // predictor-corrector labels

  // scalar diffusion coeff for divergence constraint
  d_scalDiffCoefLabel = VarLabel::create("scalDiffCoef",
				   CCVariable<double>::getTypeDescription() );

  d_scalDiffCoefSrcLabel = VarLabel::create("scalDiffCoefSrc",
				   CCVariable<double>::getTypeDescription() );




  d_enthDiffCoefLabel = VarLabel::create("enthDiffCoef",
				   CCVariable<double>::getTypeDescription() );

  d_reactscalDiffCoefLabel = VarLabel::create("reactscalDiffCoef",
				   CCVariable<double>::getTypeDescription() );




  // for corrector step
  d_uVelRhoHatLabel = VarLabel::create("uvelRhoHat",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelRhoHatLabel= VarLabel::create("vvelRhoHat",
				  SFCYVariable<double>::getTypeDescription() );
  d_wVelRhoHatLabel= VarLabel::create("wvelRhoHat",
				   SFCZVariable<double>::getTypeDescription() );

  d_uVelRhoHat_CCLabel = VarLabel::create("uvelRhoHat_CC",
					  CCVariable<double>::getTypeDescription() );
  d_vVelRhoHat_CCLabel = VarLabel::create("vvelRhoHat_CC",
					  CCVariable<double>::getTypeDescription() );
  d_wVelRhoHat_CCLabel = VarLabel::create("wvelRhoHat_CC",
					  CCVariable<double>::getTypeDescription() );
  // div constraint
  d_divConstraintLabel = VarLabel::create("divConstraint", 
				    CCVariable<double>::getTypeDescription() );


  d_pressurePredLabel = VarLabel::create("pressurePred", 
				    CCVariable<double>::getTypeDescription() );

  // enthalpy labels
  // Enthalpy 
  d_enthalpySPLabel = VarLabel::create("enthalpySP",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyEKTLabel = VarLabel::create("enthalpyEKT",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyTempLabel = VarLabel::create("enthalpyTemp",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyFELabel = VarLabel::create("enthalpyFE",
				   CCVariable<double>::getTypeDescription() );

  d_enthalpyRXNLabel = VarLabel::create("enthalpyRXN",
				   CCVariable<double>::getTypeDescription() );


  // Enthalpy Coef
  d_enthCoefSBLMLabel = VarLabel::create("enthCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Enthalpy Non Linear Src
  d_enthNonLinSrcSBLMLabel = VarLabel::create("enthNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );

  // for radiation
  d_fvtfiveINLabel = VarLabel::create("fvtfiveIN",
				    CCVariable<double>::getTypeDescription() );
  d_tfourINLabel = VarLabel::create("tfourIN",
				    CCVariable<double>::getTypeDescription() );
  d_tfiveINLabel = VarLabel::create("tfiveIN",
				    CCVariable<double>::getTypeDescription() );
  d_tnineINLabel = VarLabel::create("tnineIN",
				    CCVariable<double>::getTypeDescription() );
  d_qrgINLabel = VarLabel::create("qrgIN",
				  CCVariable<double>::getTypeDescription() );
  d_qrsINLabel = VarLabel::create("qrsIN",
				  CCVariable<double>::getTypeDescription() );

  d_absorpINLabel = VarLabel::create("absorpIN",
				    CCVariable<double>::getTypeDescription() );
  d_sootFVINLabel = VarLabel::create("sootFVIN",
				    CCVariable<double>::getTypeDescription() );

  d_abskgINLabel = VarLabel::create("abskgIN",
				    CCVariable<double>::getTypeDescription() );
  d_radiationSRCINLabel = VarLabel::create("radiationSRCIN",
				    CCVariable<double>::getTypeDescription() );

  d_radiationFluxEINLabel = VarLabel::create("radiationFluxEIN",
					     CCVariable<double>::getTypeDescription() );
  d_radiationFluxWINLabel = VarLabel::create("radiationFluxWIN",
					     CCVariable<double>::getTypeDescription() );
  d_radiationFluxNINLabel = VarLabel::create("radiationFluxNIN",
					     CCVariable<double>::getTypeDescription() );
  d_radiationFluxSINLabel = VarLabel::create("radiationFluxSIN",
					     CCVariable<double>::getTypeDescription() );
  d_radiationFluxTINLabel = VarLabel::create("radiationFluxTIN",
					     CCVariable<double>::getTypeDescription() );
  d_radiationFluxBINLabel = VarLabel::create("radiationFluxBIN",
					     CCVariable<double>::getTypeDescription() );

  d_reactscalarSRCINLabel = VarLabel::create("reactscalarSRCIN",
				    CCVariable<double>::getTypeDescription() );



  // required for scalesimilarity
  d_stressTensorCompLabel = VarLabel::create("stressTensorComp",
					     CCVariable<double>::getTypeDescription() );
  d_stressSFCXdivLabel = VarLabel::create("stressSFCXdiv",
		                             SFCXVariable<double>::getTypeDescription() );  
  d_stressSFCYdivLabel = VarLabel::create("stressSFCYdiv",
		                             SFCYVariable<double>::getTypeDescription() );  
  d_stressSFCZdivLabel = VarLabel::create("stressSFCZdiv",
		                             SFCZVariable<double>::getTypeDescription() );
  d_stressCCXdivLabel = VarLabel::create("stressCCXdiv",
		                             CCVariable<double>::getTypeDescription() );  
  d_stressCCYdivLabel = VarLabel::create("stressCCYdiv",
		                             CCVariable<double>::getTypeDescription() );  
  d_stressCCZdivLabel = VarLabel::create("stressCCZdiv",
		                             CCVariable<double>::getTypeDescription() );
  
  d_strainTensorCompLabel = VarLabel::create("strainTensorComp",
					     CCVariable<double>::getTypeDescription() );
  d_betaIJCompLabel = VarLabel::create("betaIJComp",
					     CCVariable<double>::getTypeDescription() );
  d_cbetaIJCompLabel = VarLabel::create("cbetaIJComp",
					     CCVariable<double>::getTypeDescription() );
  d_LIJCompLabel = VarLabel::create("LIJComp",
					     CCVariable<double>::getTypeDescription() );

  d_scalarFluxCompLabel = VarLabel::create("scalarFluxComp",
					     CCVariable<double>::getTypeDescription() );
  // required for dynamic procedure
  d_strainMagnitudeLabel = VarLabel::create("strainMagnitudeLabel",
					     CCVariable<double>::getTypeDescription() );
  d_strainMagnitudeMLLabel = VarLabel::create("strainMagnitudeMLLabel",
					     CCVariable<double>::getTypeDescription() );
  d_strainMagnitudeMMLabel = VarLabel::create("strainMagnitudeMMLabel",
					     CCVariable<double>::getTypeDescription() );
  d_LalphaLabel = VarLabel::create("lalphaLabel",
					     CCVariable<double>::getTypeDescription() );
  d_cbetaHATalphaLabel = VarLabel::create("cbetaHATalphaLabel",
					     CCVariable<double>::getTypeDescription() );
  d_alphaalphaLabel = VarLabel::create("alphaalphaLabel",
					     CCVariable<double>::getTypeDescription() );

  d_CsLabel = VarLabel::create("CsLabel",
			       CCVariable<double>::getTypeDescription() );
  d_deltaCsLabel = VarLabel::create("deltaCsLabel",
			       CCVariable<double>::getTypeDescription() );
  // required for odt model label
//   d_odtDataLabel = VarLabel::create("odtDataLabel",CCVariable<odtData>::getTypeDescription());

  // Runge-Kutta 3d order properties labels
  d_refDensityInterm_label = VarLabel::create("refDensityIntermLabel",
				       sum_vartype::getTypeDescription() );

  // Runge-Kutta 3d order pressure and momentum labels
  d_pressureIntermLabel = VarLabel::create("pressureInterm", 
				    CCVariable<double>::getTypeDescription() );
  d_velocityDivergenceLabel = VarLabel::create("velocityDivergence", 
				   CCVariable<double>::getTypeDescription() );
  d_vorticityXLabel = VarLabel::create("vorticityX", 
				   CCVariable<double>::getTypeDescription() );
  d_vorticityYLabel = VarLabel::create("vorticityY", 
				   CCVariable<double>::getTypeDescription() );
  d_vorticityZLabel = VarLabel::create("vorticityZ", 
				   CCVariable<double>::getTypeDescription() );
  d_vorticityLabel = VarLabel::create("vorticity", 
				   CCVariable<double>::getTypeDescription() );

  d_velDivResidualLabel = VarLabel::create("velDivResidual", 
				   CCVariable<double>::getTypeDescription() );
  d_velocityDivergenceBCLabel = VarLabel::create("velocityDivergenceBC", 
				   CCVariable<double>::getTypeDescription() );
  d_continuityResidualLabel = VarLabel::create("continuityResidual", 
				   CCVariable<double>::getTypeDescription() );


  d_InitNormLabel = VarLabel::create("initNorm",
				       max_vartype::getTypeDescription() );
  d_ScalarClippedLabel = VarLabel::create("scalarClipped",
				       max_vartype::getTypeDescription() );
  d_ReactScalarClippedLabel = VarLabel::create("reactScalarClipped",
				       max_vartype::getTypeDescription() );
  d_uVelNormLabel = VarLabel::create("uVelNorm",
				       sum_vartype::getTypeDescription() );
  d_vVelNormLabel = VarLabel::create("vVelNorm",
				       sum_vartype::getTypeDescription() );
  d_wVelNormLabel = VarLabel::create("wVelNorm",
				       sum_vartype::getTypeDescription() );
  d_rhoNormLabel = VarLabel::create("rhoNorm",
				       sum_vartype::getTypeDescription() );
  d_negativeDensityGuess_label = VarLabel::create("negativeDensityGuess",
				       sum_vartype::getTypeDescription() );
  d_negativeDensityGuessPred_label = VarLabel::create("negativeDensityGuessPred",
				       sum_vartype::getTypeDescription() );
  d_negativeDensityGuessInterm_label = VarLabel::create("negativeDensityGuessInterm",
				       sum_vartype::getTypeDescription() );
  d_negativeEKTDensityGuess_label = VarLabel::create("negativeEKTDensityGuess",
				       sum_vartype::getTypeDescription() );
  d_negativeEKTDensityGuessPred_label = VarLabel::create("negativeEKTDensityGuessPred",
				       sum_vartype::getTypeDescription() );
  d_negativeEKTDensityGuessInterm_label = VarLabel::create("negativeEKTDensityGuessInterm",
				       sum_vartype::getTypeDescription() );
  d_densityLag_label = VarLabel::create("densityLag",
				       sum_vartype::getTypeDescription() );
  d_densityLagPred_label = VarLabel::create("densityLagPred",
				       sum_vartype::getTypeDescription() );
  d_densityLagInterm_label = VarLabel::create("densityLagInterm",
				       sum_vartype::getTypeDescription() );
  d_densityLagAfterAverage_label = VarLabel::create("densityLagAfterAverage",
				       sum_vartype::getTypeDescription() );
  d_densityLagAfterIntermAverage_label = VarLabel::create("densityLagAfterIntermAverage",
				       sum_vartype::getTypeDescription() );
// kinetic energy
  d_kineticEnergyLabel = VarLabel::create("kineticEnergy", 
				   CCVariable<double>::getTypeDescription() );
  d_totalKineticEnergyLabel = VarLabel::create("totalKineticEnergy",
				       sum_vartype::getTypeDescription() );
  d_totalKineticEnergyPredLabel = VarLabel::create("totalKineticEnergyPred",
				       sum_vartype::getTypeDescription() );
  d_totalKineticEnergyIntermLabel = VarLabel::create("totalKineticEnergyInterm",
				       sum_vartype::getTypeDescription() );
// scalar mms and gradP Ln error
// ** warning...the L2 error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_smmsLnErrorLabel = VarLabel::create("smmsLnError",
					CCVariable<double>::getTypeDescription() );
  d_totalsmmsLnErrorLabel = VarLabel::create("totalsmmsLnError",
				       sum_vartype::getTypeDescription() );
  d_totalsmmsLnErrorPredLabel = VarLabel::create("totalsmmsLnErrorPred",
				       sum_vartype::getTypeDescription() );
  d_totalsmmsLnErrorIntermLabel = VarLabel::create("totalsmmsLnErrorInterm",
				       sum_vartype::getTypeDescription() );
  d_totalsmmsExactSolLabel = VarLabel::create("totalsmmsExactSol",
				       sum_vartype::getTypeDescription() );
  d_totalsmmsExactSolPredLabel = VarLabel::create("totalsmmsExactSolPred",
				       sum_vartype::getTypeDescription() );
  d_totalsmmsExactSolIntermLabel = VarLabel::create("totalsmmsExactSolInterm",
				       sum_vartype::getTypeDescription() );

  d_gradpmmsLnErrorLabel = VarLabel::create("gradpmmsLnError",
					CCVariable<double>::getTypeDescription() );
  d_totalgradpmmsLnErrorLabel = VarLabel::create("totalgradpmmsLnError",
				       sum_vartype::getTypeDescription() );
  d_totalgradpmmsLnErrorPredLabel = VarLabel::create("totalgradpmmsLnErrorPred",
				       sum_vartype::getTypeDescription() );
  d_totalgradpmmsLnErrorIntermLabel = VarLabel::create("totalgradpmmsLnErrorInterm",
				       sum_vartype::getTypeDescription() );
  d_totalgradpmmsExactSolLabel = VarLabel::create("totalgradpmmsExactSol",
				       sum_vartype::getTypeDescription() );
  d_totalgradpmmsExactSolPredLabel = VarLabel::create("totalgradpmmsExactSolPred",
				       sum_vartype::getTypeDescription() );
  d_totalgradpmmsExactSolIntermLabel = VarLabel::create("totalgradpmmsExactSolInterm",
				       sum_vartype::getTypeDescription() );


// u mms L2 error
// ** warning...the L2 error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_ummsLnErrorLabel = VarLabel::create("ummsLnError", 
				   SFCXVariable<double>::getTypeDescription() );
  d_totalummsLnErrorLabel = VarLabel::create("totalummsLnError",
				       sum_vartype::getTypeDescription() );
  d_totalummsLnErrorPredLabel = VarLabel::create("totalummsLnErrorPred",
				       sum_vartype::getTypeDescription() );
  d_totalummsLnErrorIntermLabel = VarLabel::create("totalummsLnErrorInterm",
				       sum_vartype::getTypeDescription() );
  d_totalummsExactSolLabel = VarLabel::create("totalummsExactSol",
				       sum_vartype::getTypeDescription() );
  d_totalummsExactSolPredLabel = VarLabel::create("totalummsExactSolPred",
				       sum_vartype::getTypeDescription() );
  d_totalummsExactSolIntermLabel = VarLabel::create("totalummsExactSolInterm",
				       sum_vartype::getTypeDescription() );

// v mms Ln error
// ** warning...the Ln error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_vmmsLnErrorLabel = VarLabel::create("vmmsLnError", 
				   SFCYVariable<double>::getTypeDescription() );
  d_totalvmmsLnErrorLabel = VarLabel::create("totalvmmsLnError",
				       sum_vartype::getTypeDescription() );
  d_totalvmmsLnErrorPredLabel = VarLabel::create("totalvmmsLnErrorPred",
				       sum_vartype::getTypeDescription() );
  d_totalvmmsLnErrorIntermLabel = VarLabel::create("totalvmmsLnErrorInterm",
				       sum_vartype::getTypeDescription() );
  d_totalvmmsExactSolLabel = VarLabel::create("totalvmmsExactSol",
				       sum_vartype::getTypeDescription() );
  d_totalvmmsExactSolPredLabel = VarLabel::create("totalvmmsExactSolPred",
				       sum_vartype::getTypeDescription() );
  d_totalvmmsExactSolIntermLabel = VarLabel::create("totalvmmsExactSolInterm",
				       sum_vartype::getTypeDescription() );

// w mms Ln error
// ** warning...the Ln error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_wmmsLnErrorLabel = VarLabel::create("wmmsLnError", 
				   SFCZVariable<double>::getTypeDescription() );
  d_totalwmmsLnErrorLabel = VarLabel::create("totalwmmsLnError",
				       sum_vartype::getTypeDescription() );
  d_totalwmmsLnErrorPredLabel = VarLabel::create("totalwmmsLnErrorPred",
				       sum_vartype::getTypeDescription() );
  d_totalwmmsLnErrorIntermLabel = VarLabel::create("totalwmmsLnErrorInterm",
				       sum_vartype::getTypeDescription() );
  d_totalwmmsExactSolLabel = VarLabel::create("totalwmmsExactSol",
				       sum_vartype::getTypeDescription() );
  d_totalwmmsExactSolPredLabel = VarLabel::create("totalwmmsExactSolPred",
				       sum_vartype::getTypeDescription() );
  d_totalwmmsExactSolIntermLabel = VarLabel::create("totalwmmsExactSolInterm",
				       sum_vartype::getTypeDescription() );

// mass balance labels for RK
  d_totalflowINPredLabel = VarLabel::create("totalflowINPred",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowOUTPredLabel = VarLabel::create("totalflowOUTPred",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_denAccumPredLabel = VarLabel::create("denAccumPred",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_netflowOUTBCPredLabel = VarLabel::create("netflowOUTBCPred",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalAreaOUTPredLabel = VarLabel::create("totalAreaOUTPred",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowINIntermLabel = VarLabel::create("totalflowINInterm",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalflowOUTIntermLabel = VarLabel::create("totalflowOUTInterm",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_denAccumIntermLabel = VarLabel::create("denAccumInterm",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_netflowOUTBCIntermLabel = VarLabel::create("netflowOUTBCInterm",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_totalAreaOUTIntermLabel = VarLabel::create("totalAreaOUTInterm",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 

  d_oldDeltaTLabel = VarLabel::create("oldDeltaT",
				       delt_vartype::getTypeDescription() );
// test filtered terms for variable density dynamic Smagorinsky model
  d_filterRhoULabel = VarLabel::create("filterRhoU",
				   SFCXVariable<double>::getTypeDescription() );
  d_filterRhoVLabel = VarLabel::create("filterRhoV",
				   SFCYVariable<double>::getTypeDescription() );
  d_filterRhoWLabel = VarLabel::create("filterRhoW",
				   SFCZVariable<double>::getTypeDescription() );
  d_filterRhoLabel = VarLabel::create("filterRho",
				   CCVariable<double>::getTypeDescription() );
  d_filterRhoFLabel = VarLabel::create("filterRhoF",
				   CCVariable<double>::getTypeDescription() );
  d_filterRhoELabel = VarLabel::create("filterRhoE",
				   CCVariable<double>::getTypeDescription() );
  d_filterRhoRFLabel = VarLabel::create("filterRhoRF",
				   CCVariable<double>::getTypeDescription() );
  d_scalarGradientCompLabel = VarLabel::create("scalarGradientComp",
				   CCVariable<double>::getTypeDescription() );
  d_filterScalarGradientCompLabel = VarLabel::create("filterScalarGradientComp",
				   CCVariable<double>::getTypeDescription() );
  d_enthalpyGradientCompLabel = VarLabel::create("enthalpyGradientComp",
				   CCVariable<double>::getTypeDescription() );
  d_filterEnthalpyGradientCompLabel = 
	                        VarLabel::create("filterEnthalpyGradientComp",
				CCVariable<double>::getTypeDescription() );
  d_reactScalarGradientCompLabel = VarLabel::create("reactScalarGradientComp",
				   CCVariable<double>::getTypeDescription() );
  d_filterReactScalarGradientCompLabel =
	                       VarLabel::create("filterReactScalarGradientComp",
			       CCVariable<double>::getTypeDescription() );
  d_filterStrainTensorCompLabel = VarLabel::create("filterStrainTensorComp",
			          CCVariable<double>::getTypeDescription() );
  d_scalarNumeratorLabel = VarLabel::create("scalarNumerator", 
			       CCVariable<double>::getTypeDescription() );
  d_scalarDenominatorLabel = VarLabel::create("scalarDenominator", 
			       CCVariable<double>::getTypeDescription() );
  d_enthalpyNumeratorLabel = VarLabel::create("enthalpyNumerator", 
			       CCVariable<double>::getTypeDescription() );
  d_enthalpyDenominatorLabel = VarLabel::create("enthalpyDenominator", 
			       CCVariable<double>::getTypeDescription() );
  d_reactScalarNumeratorLabel = VarLabel::create("reactScalarNumerator", 
			       CCVariable<double>::getTypeDescription() );
  d_reactScalarDenominatorLabel = VarLabel::create("reactScalarDenominator", 
			       CCVariable<double>::getTypeDescription() );
  d_ShFLabel = VarLabel::create("ShF", 
			       CCVariable<double>::getTypeDescription() );
  d_ShELabel = VarLabel::create("ShE", 
			       CCVariable<double>::getTypeDescription() );
  d_ShRFLabel = VarLabel::create("ShRF", 
			       CCVariable<double>::getTypeDescription() );
  // carbon balance labels
  d_CO2FlowRateLabel = VarLabel::create("CO2FlowRate",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_SO2FlowRateLabel = VarLabel::create("SO2FlowRate",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_carbonEfficiencyLabel = VarLabel::create("carbonEfficiency",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
  d_sulfurEfficiencyLabel = VarLabel::create("sulfurEfficiency",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
  d_scalarFlowRateLabel = VarLabel::create("scalarFlowRate",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_scalarEfficiencyLabel = VarLabel::create("scalarEfficiency",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
  d_enthalpyFlowRateLabel = VarLabel::create("enthalpyFlowRate",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_enthalpyEfficiencyLabel = VarLabel::create("enthalpyEfficiency",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
  d_totalRadSrcLabel = VarLabel::create("totalRadSrc",
     ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_normTotalRadSrcLabel = VarLabel::create("normTotalRadSrc",
	  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 

  //MMS labels
  d_uFmmsLabel = VarLabel::create("uFmms",
				   SFCXVariable<double>::getTypeDescription());
  d_vFmmsLabel = VarLabel::create("vFmms",
				   SFCYVariable<double>::getTypeDescription());
  d_wFmmsLabel = VarLabel::create("wFmms",
				   SFCZVariable<double>::getTypeDescription());


}

//****************************************************************************
// Destructor
//****************************************************************************
ArchesLabel::~ArchesLabel()
{
  if (d_stencilMatl->removeReference())
    delete d_stencilMatl;

  if (d_vectorMatl->removeReference())
    delete d_vectorMatl;

  if (d_tensorMatl->removeReference())
    delete d_tensorMatl;

  if (d_symTensorMatl->removeReference())
    delete d_symTensorMatl;

  VarLabel::destroy(d_strainMagnitudeLabel);
  VarLabel::destroy(d_strainMagnitudeMLLabel);
  VarLabel::destroy(d_strainMagnitudeMMLabel);
  VarLabel::destroy(d_LalphaLabel);
  VarLabel::destroy(d_cbetaHATalphaLabel);
  VarLabel::destroy(d_alphaalphaLabel);
  VarLabel::destroy(d_CsLabel);
  VarLabel::destroy(d_deltaCsLabel);
  VarLabel::destroy(d_cellInfoLabel);
  VarLabel::destroy(d_cellTypeLabel);
  VarLabel::destroy(d_totalflowINLabel);
  VarLabel::destroy(d_totalflowOUTLabel);
  VarLabel::destroy(d_netflowOUTBCLabel);
  VarLabel::destroy(d_totalAreaOUTLabel);
  VarLabel::destroy(d_denAccumLabel);
  VarLabel::destroy(d_densityCPLabel);
  VarLabel::destroy(d_densityEKTLabel);
  VarLabel::destroy(d_densityGuessLabel);
  VarLabel::destroy(d_densityTempLabel);
  VarLabel::destroy(d_densityOldOldLabel);
  VarLabel::destroy(d_filterdrhodtLabel);
  VarLabel::destroy(d_drhodfCPLabel);
  VarLabel::destroy(d_viscosityCTSLabel);
  VarLabel::destroy(d_scalarDiffusivityLabel);
  VarLabel::destroy(d_enthalpyDiffusivityLabel);
  VarLabel::destroy(d_reactScalarDiffusivityLabel);
  VarLabel::destroy(d_pressurePSLabel);
  VarLabel::destroy(d_pressureExtraProjectionLabel);
  VarLabel::destroy(d_presCoefPBLMLabel);
  VarLabel::destroy(d_presNonLinSrcPBLMLabel);
  VarLabel::destroy(d_uVelocitySPBCLabel);
  VarLabel::destroy(d_vVelocitySPBCLabel);
  VarLabel::destroy(d_wVelocitySPBCLabel);
  VarLabel::destroy(d_uVelocityEKTLabel);
  VarLabel::destroy(d_vVelocityEKTLabel);
  VarLabel::destroy(d_wVelocityEKTLabel);
  VarLabel::destroy(d_scalarSPLabel);
  VarLabel::destroy(d_scalarEKTLabel);
  VarLabel::destroy(d_scalarTempLabel);
  VarLabel::destroy(d_scalarFELabel);
  VarLabel::destroy(d_scalarVarSPLabel);
  VarLabel::destroy(d_scalarDissSPLabel);
  VarLabel::destroy(d_scalCoefSBLMLabel);
  VarLabel::destroy(d_scalNonLinSrcSBLMLabel);
  VarLabel::destroy(d_reactscalarSPLabel);
  VarLabel::destroy(d_reactscalarEKTLabel);
  VarLabel::destroy(d_reactscalarTempLabel);
  VarLabel::destroy(d_reactscalarFELabel);
  VarLabel::destroy(d_reactscalarVarSPLabel);
  VarLabel::destroy(d_reactscalCoefSBLMLabel);
  VarLabel::destroy(d_reactscalNonLinSrcSBLMLabel);
  VarLabel::destroy(d_refDensity_label);
  VarLabel::destroy(d_refDensityPred_label);
  VarLabel::destroy(d_refPressure_label);
  VarLabel::destroy(d_oldCCVelocityLabel);
  VarLabel::destroy(d_newCCVelocityLabel);
  VarLabel::destroy(d_newCCVelMagLabel);
  VarLabel::destroy(d_newCCUVelocityLabel);
  VarLabel::destroy(d_newCCVVelocityLabel);
  VarLabel::destroy(d_newCCWVelocityLabel);
  VarLabel::destroy(d_mmcellTypeLabel);
  VarLabel::destroy(d_mmgasVolFracLabel);
  VarLabel::destroy(d_tempINLabel);
  VarLabel::destroy(d_cpINLabel);
  VarLabel::destroy(d_co2INLabel);
  VarLabel::destroy(d_h2oINLabel);
  VarLabel::destroy(d_normalizedScalarVarLabel);
  VarLabel::destroy(d_heatLossLabel);
  VarLabel::destroy(d_h2sINLabel);
  VarLabel::destroy(d_so2INLabel);
  VarLabel::destroy(d_so3INLabel);
  VarLabel::destroy(d_sulfurINLabel);
  VarLabel::destroy(d_coINLabel);
  VarLabel::destroy(d_c2h2INLabel);
  VarLabel::destroy(d_ch4INLabel);
  VarLabel::destroy(d_denRefArrayLabel);
  VarLabel::destroy(d_densityMicroLabel);
  VarLabel::destroy(d_densityMicroINLabel);
  VarLabel::destroy(d_pressPlusHydroLabel);
  VarLabel::destroy(d_uvwoutLabel);
  VarLabel::destroy(d_scalDiffCoefLabel);
  VarLabel::destroy(d_scalDiffCoefSrcLabel);
  VarLabel::destroy(d_enthDiffCoefLabel);
  VarLabel::destroy(d_reactscalDiffCoefLabel);
  VarLabel::destroy(d_uVelRhoHatLabel);
  VarLabel::destroy(d_vVelRhoHatLabel);
  VarLabel::destroy(d_wVelRhoHatLabel);
  VarLabel::destroy(d_uVelRhoHat_CCLabel);
  VarLabel::destroy(d_vVelRhoHat_CCLabel);
  VarLabel::destroy(d_wVelRhoHat_CCLabel);
  VarLabel::destroy(d_divConstraintLabel); 
  VarLabel::destroy(d_pressurePredLabel);
  VarLabel::destroy(d_enthalpySPLabel);
  VarLabel::destroy(d_enthalpyEKTLabel);
  VarLabel::destroy(d_enthalpyTempLabel);
  VarLabel::destroy(d_enthalpyFELabel);
  VarLabel::destroy(d_enthalpyRXNLabel);
  VarLabel::destroy(d_enthCoefSBLMLabel);
  VarLabel::destroy(d_enthNonLinSrcSBLMLabel);
  VarLabel::destroy(d_absorpINLabel);
  VarLabel::destroy(d_fvtfiveINLabel);
  VarLabel::destroy(d_tfourINLabel);
  VarLabel::destroy(d_tfiveINLabel);
  VarLabel::destroy(d_tnineINLabel);
  VarLabel::destroy(d_qrgINLabel);
  VarLabel::destroy(d_qrsINLabel);
  VarLabel::destroy(d_abskgINLabel);
  VarLabel::destroy(d_sootFVINLabel);
  VarLabel::destroy(d_radiationSRCINLabel);
  VarLabel::destroy(d_radiationFluxEINLabel);
  VarLabel::destroy(d_radiationFluxWINLabel);
  VarLabel::destroy(d_radiationFluxNINLabel);
  VarLabel::destroy(d_radiationFluxSINLabel);
  VarLabel::destroy(d_radiationFluxTINLabel);
  VarLabel::destroy(d_radiationFluxBINLabel);
  VarLabel::destroy(d_reactscalarSRCINLabel);
 // Runge-Kutta 3d order properties labels
  VarLabel::destroy(d_refDensityInterm_label);
 // Runge-Kutta 3d order pressure and momentum labels
  VarLabel::destroy(d_pressureIntermLabel);
// labels for scale similarity model
  VarLabel::destroy(d_stressTensorCompLabel);
  VarLabel::destroy(d_stressSFCXdivLabel);  
  VarLabel::destroy(d_stressSFCYdivLabel);
  VarLabel::destroy(d_stressSFCZdivLabel); 
  VarLabel::destroy(d_stressCCXdivLabel);  
  VarLabel::destroy(d_stressCCYdivLabel);
  VarLabel::destroy(d_stressCCZdivLabel); 
  
  VarLabel::destroy(d_strainTensorCompLabel);
  VarLabel::destroy(d_betaIJCompLabel);
  VarLabel::destroy(d_cbetaIJCompLabel);
  VarLabel::destroy(d_LIJCompLabel);
  VarLabel::destroy(d_scalarFluxCompLabel);
  VarLabel::destroy(d_velocityDivergenceLabel);
  VarLabel::destroy(d_vorticityXLabel);
  VarLabel::destroy(d_vorticityYLabel);
  VarLabel::destroy(d_vorticityZLabel);
  VarLabel::destroy(d_vorticityLabel);
  VarLabel::destroy(d_velDivResidualLabel);
  VarLabel::destroy(d_velocityDivergenceBCLabel);
  VarLabel::destroy(d_continuityResidualLabel);

  VarLabel::destroy(d_InitNormLabel);
  VarLabel::destroy(d_ScalarClippedLabel);
  VarLabel::destroy(d_ReactScalarClippedLabel);
  VarLabel::destroy(d_uVelNormLabel);
  VarLabel::destroy(d_vVelNormLabel);
  VarLabel::destroy(d_wVelNormLabel);
  VarLabel::destroy(d_rhoNormLabel);

// label for odt model
  //VarLabel::destroy(d_odtDataLabel);

  VarLabel::destroy(d_negativeDensityGuess_label);
  VarLabel::destroy(d_negativeDensityGuessPred_label);
  VarLabel::destroy(d_negativeDensityGuessInterm_label);
  VarLabel::destroy(d_negativeEKTDensityGuess_label);
  VarLabel::destroy(d_negativeEKTDensityGuessPred_label);
  VarLabel::destroy(d_negativeEKTDensityGuessInterm_label);
  VarLabel::destroy(d_densityLag_label);
  VarLabel::destroy(d_densityLagPred_label);
  VarLabel::destroy(d_densityLagInterm_label);
  VarLabel::destroy(d_densityLagAfterAverage_label);
  VarLabel::destroy(d_densityLagAfterIntermAverage_label);
// kinetic energy
  VarLabel::destroy(d_kineticEnergyLabel); 
  VarLabel::destroy(d_totalKineticEnergyLabel); 
  VarLabel::destroy(d_totalKineticEnergyPredLabel); 
  VarLabel::destroy(d_totalKineticEnergyIntermLabel); 
// mass balance labels for RK
  VarLabel::destroy(d_totalflowINPredLabel);
  VarLabel::destroy(d_totalflowOUTPredLabel);
  VarLabel::destroy(d_denAccumPredLabel);
  VarLabel::destroy(d_netflowOUTBCPredLabel);
  VarLabel::destroy(d_totalAreaOUTPredLabel);
  VarLabel::destroy(d_totalflowINIntermLabel);
  VarLabel::destroy(d_totalflowOUTIntermLabel);
  VarLabel::destroy(d_denAccumIntermLabel);
  VarLabel::destroy(d_netflowOUTBCIntermLabel);
  VarLabel::destroy(d_totalAreaOUTIntermLabel);
      	       
  VarLabel::destroy(d_oldDeltaTLabel);

// test filtered terms for variable density dynamic Smagorinsky model
  VarLabel::destroy(d_filterRhoULabel);
  VarLabel::destroy(d_filterRhoVLabel);
  VarLabel::destroy(d_filterRhoWLabel);
  VarLabel::destroy(d_filterRhoLabel);
  VarLabel::destroy(d_filterRhoFLabel);
  VarLabel::destroy(d_filterRhoELabel);
  VarLabel::destroy(d_filterRhoRFLabel);
  VarLabel::destroy(d_scalarGradientCompLabel);
  VarLabel::destroy(d_filterScalarGradientCompLabel);
  VarLabel::destroy(d_enthalpyGradientCompLabel);
  VarLabel::destroy(d_filterEnthalpyGradientCompLabel);
  VarLabel::destroy(d_reactScalarGradientCompLabel);
  VarLabel::destroy(d_filterReactScalarGradientCompLabel);
  VarLabel::destroy(d_filterStrainTensorCompLabel);
  VarLabel::destroy(d_scalarNumeratorLabel); 
  VarLabel::destroy(d_scalarDenominatorLabel); 
  VarLabel::destroy(d_enthalpyNumeratorLabel); 
  VarLabel::destroy(d_enthalpyDenominatorLabel); 
  VarLabel::destroy(d_reactScalarNumeratorLabel); 
  VarLabel::destroy(d_reactScalarDenominatorLabel); 
  VarLabel::destroy(d_ShFLabel);
  VarLabel::destroy(d_ShELabel);
  VarLabel::destroy(d_ShRFLabel);
  VarLabel::destroy(d_CO2FlowRateLabel);
  VarLabel::destroy(d_carbonEfficiencyLabel);
  VarLabel::destroy(d_SO2FlowRateLabel);
  VarLabel::destroy(d_sulfurEfficiencyLabel);
  VarLabel::destroy(d_scalarFlowRateLabel);
  VarLabel::destroy(d_scalarEfficiencyLabel);
  VarLabel::destroy(d_enthalpyFlowRateLabel);
  VarLabel::destroy(d_enthalpyEfficiencyLabel);
  VarLabel::destroy(d_totalRadSrcLabel);
  VarLabel::destroy(d_normTotalRadSrcLabel);

  //mms variabels
  VarLabel::destroy(d_uFmmsLabel);
  VarLabel::destroy(d_vFmmsLabel);
  VarLabel::destroy(d_wFmmsLabel);

  VarLabel::destroy(d_ummsLnErrorLabel);
  VarLabel::destroy(d_totalummsLnErrorLabel);
  VarLabel::destroy(d_totalummsLnErrorPredLabel);
  VarLabel::destroy(d_totalummsLnErrorIntermLabel);
  VarLabel::destroy(d_totalummsExactSolLabel);
  VarLabel::destroy(d_totalummsExactSolPredLabel);
  VarLabel::destroy(d_totalummsExactSolIntermLabel);

  VarLabel::destroy(d_vmmsLnErrorLabel);
  VarLabel::destroy(d_totalvmmsLnErrorLabel);
  VarLabel::destroy(d_totalvmmsLnErrorPredLabel);
  VarLabel::destroy(d_totalvmmsLnErrorIntermLabel);
  VarLabel::destroy(d_totalvmmsExactSolLabel);
  VarLabel::destroy(d_totalvmmsExactSolPredLabel);
  VarLabel::destroy(d_totalvmmsExactSolIntermLabel);

  VarLabel::destroy(d_wmmsLnErrorLabel);
  VarLabel::destroy(d_totalwmmsLnErrorLabel);
  VarLabel::destroy(d_totalwmmsLnErrorPredLabel);
  VarLabel::destroy(d_totalwmmsLnErrorIntermLabel);
  VarLabel::destroy(d_totalwmmsExactSolLabel);
  VarLabel::destroy(d_totalwmmsExactSolPredLabel);
  VarLabel::destroy(d_totalwmmsExactSolIntermLabel);

  VarLabel::destroy(d_smmsLnErrorLabel);
  VarLabel::destroy(d_totalsmmsLnErrorLabel);
  VarLabel::destroy(d_totalsmmsLnErrorPredLabel);
  VarLabel::destroy(d_totalsmmsLnErrorIntermLabel);
  VarLabel::destroy(d_totalsmmsExactSolLabel);
  VarLabel::destroy(d_totalsmmsExactSolPredLabel);
  VarLabel::destroy(d_totalsmmsExactSolIntermLabel);

  VarLabel::destroy(d_gradpmmsLnErrorLabel);
  VarLabel::destroy(d_totalgradpmmsLnErrorLabel);
  VarLabel::destroy(d_totalgradpmmsLnErrorPredLabel);
  VarLabel::destroy(d_totalgradpmmsLnErrorIntermLabel);
  VarLabel::destroy(d_totalgradpmmsExactSolLabel);
  VarLabel::destroy(d_totalgradpmmsExactSolPredLabel);
  VarLabel::destroy(d_totalgradpmmsExactSolIntermLabel);
  
}           

void ArchesLabel::setSharedState(SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
}
