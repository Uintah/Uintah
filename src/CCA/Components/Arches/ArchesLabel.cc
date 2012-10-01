/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- ArchesLabel.cc ----------------------------------------------

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for ArchesLabel
//****************************************************************************
ArchesLabel::ArchesLabel()
{

  // These are allowed CFD roles: 
  d_allowed_roles.push_back("temperature"); 
  d_allowed_roles.push_back("density");
  d_allowed_roles.push_back("enthalpy"); 

   // shortcuts
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  const TypeDescription* CC_Vector = CCVariable<Vector>::getTypeDescription();

  const TypeDescription* SFCX_double = SFCXVariable<double>::getTypeDescription();
  const TypeDescription* SFCY_double = SFCYVariable<double>::getTypeDescription();
  const TypeDescription* SFCZ_double = SFCZVariable<double>::getTypeDescription();

  const TypeDescription* sum_variable = sum_vartype::getTypeDescription();
  const TypeDescription* max_variable = max_vartype::getTypeDescription();
  const TypeDescription* min_variable = min_vartype::getTypeDescription();

  // Seven point stencil
  int numberStencilComponents = 7;
  d_stencilMatl = scinew MaterialSubset();
  for (int i = 0; i < numberStencilComponents; i++){
    d_stencilMatl->add(i);
  }
  d_stencilMatl->addReference();

  // Vector (1 order tensor)
  int numberVectorComponents = 3;
  d_vectorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberVectorComponents; i++){
    d_vectorMatl->add(i);
  }
  d_vectorMatl->addReference();

  // Second order tensor
  int numberTensorComponents = 9;
  d_tensorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberTensorComponents; i++){
    d_tensorMatl->add(i);
  }
  d_tensorMatl->addReference();

  // Second order symmetric tensor
  int numberSymTensorComponents = 6;
  d_symTensorMatl = scinew MaterialSubset();
  for (int i = 0; i < numberSymTensorComponents; i++){
    d_symTensorMatl->add(i);
  }
  d_symTensorMatl->addReference();

  // Cell Information
  d_cellInfoLabel = VarLabel::create("cellInformation",
                            PerPatch<CellInformationP>::getTypeDescription());
  // Cell type
  d_cellTypeLabel = VarLabel::create("cellType", 
                                  CCVariable<int>::getTypeDescription());
  // labels for inlet and outlet flow rates
  d_totalflowINLabel   =  VarLabel::create("totalflowIN",   sum_variable);
  d_totalflowOUTLabel  =  VarLabel::create("totalflowOUT",  sum_variable);
  d_netflowOUTBCLabel  =  VarLabel::create("netflowOUTBC",  sum_variable);
  d_totalAreaOUTLabel  =  VarLabel::create("totalAreaOUT",  sum_variable);
  d_denAccumLabel      =  VarLabel::create("denAccum",      sum_variable);


  // Density Labels
  d_densityCPLabel      =  VarLabel::create("densityCP",      CC_double);
  d_densityGuessLabel   =  VarLabel::create("densityGuess",   CC_double);
  d_densityTempLabel    =  VarLabel::create("densityTemp",    CC_double);
  d_densityOldOldLabel  =  VarLabel::create("densityOldOld",  CC_double);
  d_filterdrhodtLabel   =  VarLabel::create("filterdrhodt",   CC_double);
  d_drhodfCPLabel       =  VarLabel::create("drhodfCP",       CC_double);

  
  // Viscosity Labels
  d_viscosityCTSLabel           = VarLabel::create("viscosityCTS",            CC_double);
  d_tauSGSLabel                 = VarLabel::create("tauSGS",                  CC_double);
  d_scalarDiffusivityLabel      = VarLabel::create("scalarDiffusivity",       CC_double);
  d_enthalpyDiffusivityLabel    = VarLabel::create("enthalpyDiffusivity",     CC_double);
  d_reactScalarDiffusivityLabel = VarLabel::create("reactScalarDiffusivity",  CC_double);
  
  // Pressure Labels
  d_pressurePSLabel    = VarLabel::create("pressurePS", CC_double);
  d_pressureGuessLabel = VarLabel::create("pressureGuess", CC_double);
  d_pressureExtraProjectionLabel = VarLabel::create("pressureExtraProjection", CC_double);
  
  
  // Pressure Coeff Labels
  d_presCoefPBLMLabel      = VarLabel::create("presCoefPBLM", CCVariable<Stencil7>::getTypeDescription());
  // Pressure Non Linear Src Labels
  d_presNonLinSrcPBLMLabel = VarLabel::create("presNonLinSrcPBLM", CC_double);
  
  // Velocity Labels
  d_uVelocitySPBCLabel = VarLabel::create("uVelocitySPBC", SFCX_double);
  d_vVelocitySPBCLabel = VarLabel::create("vVelocitySPBC", SFCY_double);
  d_wVelocitySPBCLabel = VarLabel::create("wVelocitySPBC", SFCZ_double);
  
  // labels for ref density and pressure
  d_refDensity_label      =  VarLabel::create("refDensityLabel",      sum_variable);
  d_refDensityPred_label  =  VarLabel::create("refDensityPredLabel",  sum_variable);
  d_refPressurePred_label =  VarLabel::create("refPressurePredLabel", sum_variable);
  d_refPressure_label     =  VarLabel::create("refPressureLabel",     sum_variable);

  // Scalar 
  d_scalarSPLabel    =  VarLabel::create("scalarSP",    CC_double);
  d_scalarTempLabel  =  VarLabel::create("scalarTemp",  CC_double);
  d_scalarFELabel    =  VarLabel::create("scalarFE",    CC_double);

  // scalar variance
  d_scalarVarSPLabel  = VarLabel::create("scalarVarSP",  CC_double);
  d_scalarDissSPLabel = VarLabel::create("scalarDissSP", CC_double);
  
  // Scalar Coef
  d_scalCoefSBLMLabel = VarLabel::create("scalCoefSBLM",CC_double);

  // New Scalar Coefs
  d_scalarTotCoefLabel  = VarLabel::create("scalarTotCoef",  CC_double); 

  // Scalar Non Linear Src
  d_scalNonLinSrcSBLMLabel = VarLabel::create("scalNonLinSrcSBLM",CC_double);

  // reactive scalar
  d_reactscalarSPLabel    =  VarLabel::create("reactscalarSP",    CC_double);
  d_reactscalarTempLabel  =  VarLabel::create("reactscalarTemp",  CC_double);
  d_reactscalarFELabel    =  VarLabel::create("reactscalarFE",    CC_double);

  // Reactscalar Coef
  d_reactscalCoefSBLMLabel = VarLabel::create("reactscalCoefSBLM",  CC_double);
  // Reactscalar Non Linear Src
  d_reactscalNonLinSrcSBLMLabel = VarLabel::create("reactscalNonLinSrcSBLM",CC_double);

  // Cell Centered data after interpolation (for use in visualization and turbulence models)
  d_CCVelocityLabel   =  VarLabel::create("CCVelocity",      CC_Vector);

  // multimaterial wall/intrusion cells
  d_mmcellTypeLabel = VarLabel::create("mmcellType",
                                      CCVariable<int>::getTypeDescription());

  // Label for void fraction, after correction for wall cells using cutoff
  d_mmgasVolFracLabel = VarLabel::create("mmgasVolFrac",  CC_double);

  // for reacting flows
  d_normalizedScalarVarLabel = VarLabel::create("normalizedScalarVar",CC_double);
  d_dummyTLabel    =  VarLabel::create("dummyT",    CC_double);
  d_tempINLabel    =  VarLabel::create("tempIN",    CC_double);
  d_tempFxLabel    =  VarLabel::create("tempFxLabel", SFCX_double); 
  d_tempFyLabel    =  VarLabel::create("tempFyLabel", SFCY_double); 
  d_tempFzLabel    =  VarLabel::create("tempFzLabel", SFCZ_double); 
  d_cpINLabel      =  VarLabel::create("cpIN",      CC_double);
  d_co2INLabel     =  VarLabel::create("co2IN",     CC_double);
  d_heatLossLabel  =  VarLabel::create("heatLoss",  CC_double);
  d_h2oINLabel     =  VarLabel::create("h2oIN",     CC_double);
  d_h2sINLabel     =  VarLabel::create("h2sIN",     CC_double);
  d_so2INLabel     =  VarLabel::create("so2IN",     CC_double);
  d_so3INLabel     =  VarLabel::create("so3IN",     CC_double);
  d_sulfurINLabel  =  VarLabel::create("sulfurIN",  CC_double);
  d_s2INLabel      =  VarLabel::create("s2IN",      CC_double);
  d_shINLabel      =  VarLabel::create("shIN",      CC_double);
  d_soINLabel      =  VarLabel::create("soIN",      CC_double);
  d_hso2INLabel    =  VarLabel::create("hso2IN",    CC_double);
  d_hosoINLabel    =  VarLabel::create("hosoIN",    CC_double);
  d_hoso2INLabel   =  VarLabel::create("hoso2IN",   CC_double);
  d_snINLabel      =  VarLabel::create("snIN",      CC_double);
  d_csINLabel      =  VarLabel::create("csIN",      CC_double);
  d_ocsINLabel     =  VarLabel::create("ocsIN",     CC_double);
  d_hsoINLabel     =  VarLabel::create("hsoIN",     CC_double);
  d_hosINLabel     =  VarLabel::create("hosIN",     CC_double);
  d_hsohINLabel    =  VarLabel::create("hsohIN",    CC_double);
  d_h2soINLabel    =  VarLabel::create("h2soIN",    CC_double);
  d_hoshoINLabel   =  VarLabel::create("hoshoIN",   CC_double);
  d_hs2INLabel     =  VarLabel::create("hs2IN",     CC_double);
  d_h2s2INLabel    =  VarLabel::create("h2s2IN",    CC_double);
  d_coINLabel      =  VarLabel::create("coIN",      CC_double);
  d_c2h2INLabel    =  VarLabel::create("c2h2IN",    CC_double);
  d_ch4INLabel     =  VarLabel::create("ch4IN",     CC_double);
  d_mixMWLabel     =  VarLabel::create("mixMW",     CC_double); 

  // Array containing the reference density multiplied by the void fraction
  // used for correct reference density subtraction in the multimaterial
  // case

  d_denRefArrayLabel = VarLabel::create("denRefArray",  CC_double);

  // Microscopic density (i.e., without void fraction) of gas
  d_densityMicroLabel    =  VarLabel::create("denMicro",    CC_double);
  d_densityMicroINLabel  =  VarLabel::create("denMicroIN",  CC_double);

  // Sum of the relative pressure and the hydrostatic contribution
  d_pressPlusHydroLabel = VarLabel::create("pPlusHydro",  CC_double);


  d_uvwoutLabel = VarLabel::create("uvwout", min_variable); 


  // predictor-corrector labels

  // scalar diffusion coeff for divergence constraint
  d_scalDiffCoefLabel       =  VarLabel::create("scalDiffCoef",       CC_double);
  d_scalDiffCoefSrcLabel    =  VarLabel::create("scalDiffCoefSrc",    CC_double);
  d_enthDiffCoefLabel       =  VarLabel::create("enthDiffCoef",       CC_double);
  d_reactscalDiffCoefLabel  =  VarLabel::create("reactscalDiffCoef",  CC_double);

  // for corrector step
  d_uVelRhoHatLabel     =  VarLabel::create("uvelRhoHat",     SFCX_double);
  d_vVelRhoHatLabel     =  VarLabel::create("vvelRhoHat",     SFCY_double);
  d_wVelRhoHatLabel     =  VarLabel::create("wvelRhoHat",     SFCZ_double);
  d_uVelRhoHat_CCLabel  =  VarLabel::create("uvelRhoHat_CC",  CC_double );
  d_vVelRhoHat_CCLabel  =  VarLabel::create("vvelRhoHat_CC",  CC_double );
  d_wVelRhoHat_CCLabel  =  VarLabel::create("wvelRhoHat_CC",  CC_double );

  // div constraint
  d_divConstraintLabel = VarLabel::create("divConstraint",  CC_double);
  d_pressurePredLabel = VarLabel::create("pressurePred",    CC_double);

  // enthalpy labels
  d_enthalpySPLabel    =  VarLabel::create("enthalpySP",    CC_double);
  d_enthalpyTempLabel  =  VarLabel::create("enthalpyTemp",  CC_double);
  d_enthalpyFELabel    =  VarLabel::create("enthalpyFE",    CC_double);
  d_enthalpyRXNLabel   =  VarLabel::create("enthalpyRXN",   CC_double);
 
  // Enthalpy Coef
  d_enthCoefSBLMLabel = VarLabel::create("enthCoefSBLM",      CC_double);
  // Enthalpy Non Linear Src
  d_enthNonLinSrcSBLMLabel = VarLabel::create("enthNonLinSrcSBLM",CC_double);

  //__________________________________
  // Radiation
  d_absorpINLabel   =  VarLabel::create("absorpIN",   CC_double);
  d_sootFVINLabel   =  VarLabel::create("sootFVIN",   CC_double);
  d_abskgINLabel    =  VarLabel::create("abskgIN",    CC_double);

  d_radiationSRCINLabel = VarLabel::create("radiationSRCIN",  CC_double);

  d_radiationFluxEINLabel = VarLabel::create("radiationFluxEIN",  CC_double);
  d_radiationFluxWINLabel = VarLabel::create("radiationFluxWIN",  CC_double);
  d_radiationFluxNINLabel = VarLabel::create("radiationFluxNIN",  CC_double);
  d_radiationFluxSINLabel = VarLabel::create("radiationFluxSIN",  CC_double);
  d_radiationFluxTINLabel = VarLabel::create("radiationFluxTIN",  CC_double);
  d_radiationFluxBINLabel = VarLabel::create("radiationFluxBIN",  CC_double);
  d_radiationVolqINLabel = VarLabel::create("radiationVolqIN",  CC_double);

  d_reactscalarSRCINLabel = VarLabel::create("reactscalarSRCIN",  CC_double);    

  //__________________________________
  // Scalesimilarity
  d_stressTensorCompLabel  =  VarLabel::create("stressTensorComp",  CC_double ); 
  
  d_strainTensorCompLabel  =  VarLabel::create("strainTensorComp",  CC_double);
  d_LIJCompLabel           =  VarLabel::create("LIJComp",           CC_double);
  d_scalarFluxCompLabel    = VarLabel::create("scalarFluxComp",     CC_double);
  
  //__________________________________
  // Dynamic procedure
  d_strainMagnitudeLabel    =  VarLabel::create("strainMagnitudeLabel",    CC_double);
  d_strainMagnitudeMLLabel  =  VarLabel::create("strainMagnitudeMLLabel",  CC_double);
  d_strainMagnitudeMMLabel  =  VarLabel::create("strainMagnitudeMMLabel",  CC_double);
  d_LalphaLabel             =  VarLabel::create("lalphaLabel",             CC_double);
  d_cbetaHATalphaLabel      =  VarLabel::create("cbetaHATalphaLabel",      CC_double);
  d_alphaalphaLabel         =  VarLabel::create("alphaalphaLabel",         CC_double);
  d_CsLabel                 =  VarLabel::create("CsLabel",                 CC_double);

  // required for odt model label
//   d_odtDataLabel = VarLabel::create("odtDataLabel",CCVariable<odtData>::getTypeDescription());

  // Runge-Kutta 3d order properties labels
  d_refDensityInterm_label  = VarLabel::create("refDensityIntermLabel",  sum_variable);
  d_refPressureInterm_label = VarLabel::create("refPressureIntermLabel", sum_variable);

  // Runge-Kutta 3d order pressure and momentum labels
  d_pressureIntermLabel     = VarLabel::create("pressureInterm",     CC_double);
  d_velocityDivergenceLabel = VarLabel::create("velocityDivergence", CC_double);
  
  d_vorticityXLabel  =  VarLabel::create("vorticityX",  CC_double);
  d_vorticityYLabel  =  VarLabel::create("vorticityY",  CC_double);
  d_vorticityZLabel  =  VarLabel::create("vorticityZ",  CC_double);
  d_vorticityLabel   =  VarLabel::create("vorticity",   CC_double);

  d_velDivResidualLabel        =  VarLabel::create("velDivResidual",        CC_double);
  d_continuityResidualLabel    =  VarLabel::create("continuityResidual",    CC_double);

  d_ScalarClippedLabel       =  VarLabel::create("scalarClipped",       max_variable);
  d_ReactScalarClippedLabel  =  VarLabel::create("reactScalarClipped",  max_variable);

  d_uVelNormLabel  =  VarLabel::create("uVelNorm",  sum_variable);
  d_vVelNormLabel  =  VarLabel::create("vVelNorm",  sum_variable);
  d_wVelNormLabel  =  VarLabel::create("wVelNorm",  sum_variable);
  d_rhoNormLabel   =  VarLabel::create("rhoNorm",   sum_variable);

  d_negativeDensityGuess_label           =  VarLabel::create("negativeDensityGuess",           sum_variable);
  d_negativeDensityGuessPred_label       =  VarLabel::create("negativeDensityGuessPred",       sum_variable);
  d_negativeDensityGuessInterm_label     =  VarLabel::create("negativeDensityGuessInterm",     sum_variable);
  d_densityLag_label                     =  VarLabel::create("densityLag",                     sum_variable);
  d_densityLagPred_label                 =  VarLabel::create("densityLagPred",                 sum_variable);
  d_densityLagInterm_label               =  VarLabel::create("densityLagInterm",               sum_variable);
  d_densityLagAfterAverage_label         =  VarLabel::create("densityLagAfterAverage",         sum_variable);
  d_densityLagAfterIntermAverage_label   =  VarLabel::create("densityLagAfterIntermAverage",   sum_variable);
  
// kinetic energy
  d_kineticEnergyLabel             =  VarLabel::create("kineticEnergy",             CC_double);
  d_totalKineticEnergyLabel        =  VarLabel::create("totalKineticEnergy",        sum_variable);
  d_totalKineticEnergyPredLabel    =  VarLabel::create("totalKineticEnergyPred",    sum_variable);
  d_totalKineticEnergyIntermLabel  =  VarLabel::create("totalKineticEnergyInterm",  sum_variable);

// scalar mms and gradP Ln error
// ** warning...the L2 error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_smmsLnErrorLabel              =  VarLabel::create("smmsLnError",              CC_double);
  d_totalsmmsLnErrorLabel         =  VarLabel::create("totalsmmsLnError",         sum_variable);
  d_totalsmmsLnErrorPredLabel     =  VarLabel::create("totalsmmsLnErrorPred",     sum_variable);
  d_totalsmmsLnErrorIntermLabel   =  VarLabel::create("totalsmmsLnErrorInterm",   sum_variable);
  d_totalsmmsExactSolLabel        =  VarLabel::create("totalsmmsExactSol",        sum_variable);
  d_totalsmmsExactSolPredLabel    =  VarLabel::create("totalsmmsExactSolPred",    sum_variable);
  d_totalsmmsExactSolIntermLabel  =  VarLabel::create("totalsmmsExactSolInterm",  sum_variable);

  d_gradpmmsLnErrorLabel              =  VarLabel::create("gradpmmsLnError",              CC_double);
  d_totalgradpmmsLnErrorLabel         =  VarLabel::create("totalgradpmmsLnError",         sum_variable);
  d_totalgradpmmsLnErrorPredLabel     =  VarLabel::create("totalgradpmmsLnErrorPred",     sum_variable);
  d_totalgradpmmsLnErrorIntermLabel   =  VarLabel::create("totalgradpmmsLnErrorInterm",   sum_variable);
  d_totalgradpmmsExactSolLabel        =  VarLabel::create("totalgradpmmsExactSol",        sum_variable);
  d_totalgradpmmsExactSolPredLabel    =  VarLabel::create("totalgradpmmsExactSolPred",    sum_variable);
  d_totalgradpmmsExactSolIntermLabel  =  VarLabel::create("totalgradpmmsExactSolInterm",  sum_variable);

// u mms L2 error
// ** warning...the L2 error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_ummsLnErrorLabel              =  VarLabel::create("ummsLnError",              SFCX_double);
  d_totalummsLnErrorLabel         =  VarLabel::create("totalummsLnError",         sum_variable);
  d_totalummsLnErrorPredLabel     =  VarLabel::create("totalummsLnErrorPred",     sum_variable);
  d_totalummsLnErrorIntermLabel   =  VarLabel::create("totalummsLnErrorInterm",   sum_variable);
  d_totalummsExactSolLabel        =  VarLabel::create("totalummsExactSol",        sum_variable);
  d_totalummsExactSolPredLabel    =  VarLabel::create("totalummsExactSolPred",    sum_variable);
  d_totalummsExactSolIntermLabel  =  VarLabel::create("totalummsExactSolInterm",  sum_variable);

// v mms Ln error
// ** warning...the Ln error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_vmmsLnErrorLabel              =  VarLabel::create("vmmsLnError",              SFCY_double);
  d_totalvmmsLnErrorLabel         =  VarLabel::create("totalvmmsLnError",         sum_variable);
  d_totalvmmsLnErrorPredLabel     =  VarLabel::create("totalvmmsLnErrorPred",     sum_variable);
  d_totalvmmsLnErrorIntermLabel   =  VarLabel::create("totalvmmsLnErrorInterm",   sum_variable);
  d_totalvmmsExactSolLabel        =  VarLabel::create("totalvmmsExactSol",        sum_variable);
  d_totalvmmsExactSolPredLabel    =  VarLabel::create("totalvmmsExactSolPred",    sum_variable);
  d_totalvmmsExactSolIntermLabel  =  VarLabel::create("totalvmmsExactSolInterm",  sum_variable);


// w mms Ln error
// ** warning...the Ln error here is not complete
//              the values are (exact-comput.)^2
//              You must post process the squareroot
//              because of the summation. 
//              Alternatively, one could add an 
//              additional reduction var. and do this inline
//              with the code.
  d_wmmsLnErrorLabel              =  VarLabel::create("wmmsLnError",              SFCZ_double);
  d_totalwmmsLnErrorLabel         =  VarLabel::create("totalwmmsLnError",         sum_variable);
  d_totalwmmsLnErrorPredLabel     =  VarLabel::create("totalwmmsLnErrorPred",     sum_variable);
  d_totalwmmsLnErrorIntermLabel   =  VarLabel::create("totalwmmsLnErrorInterm",   sum_variable);
  d_totalwmmsExactSolLabel        =  VarLabel::create("totalwmmsExactSol",        sum_variable);
  d_totalwmmsExactSolPredLabel    =  VarLabel::create("totalwmmsExactSolPred",    sum_variable);
  d_totalwmmsExactSolIntermLabel  =  VarLabel::create("totalwmmsExactSolInterm",  sum_variable);

// mass balance labels for RK
  d_totalflowINPredLabel     =  VarLabel::create("totalflowINPred",     sum_variable);
  d_totalflowOUTPredLabel    =  VarLabel::create("totalflowOUTPred",    sum_variable);
  d_denAccumPredLabel        =  VarLabel::create("denAccumPred",        sum_variable);
  d_netflowOUTBCPredLabel    =  VarLabel::create("netflowOUTBCPred",    sum_variable);
  d_totalAreaOUTPredLabel    =  VarLabel::create("totalAreaOUTPred",    sum_variable);
  d_totalflowINIntermLabel   =  VarLabel::create("totalflowINInterm",   sum_variable);
  d_totalflowOUTIntermLabel  =  VarLabel::create("totalflowOUTInterm",  sum_variable);
  d_denAccumIntermLabel      =  VarLabel::create("denAccumInterm",      sum_variable);
  d_netflowOUTBCIntermLabel  =  VarLabel::create("netflowOUTBCInterm",  sum_variable);
  d_totalAreaOUTIntermLabel  =  VarLabel::create("totalAreaOUTInterm",  sum_variable);


  d_oldDeltaTLabel = VarLabel::create("oldDeltaT",
                                       delt_vartype::getTypeDescription());
// test filtered terms for variable density dynamic Smagorinsky model
  d_filterRhoULabel   =  VarLabel::create("filterRhoU",   SFCX_double);
  d_filterRhoVLabel   =  VarLabel::create("filterRhoV",   SFCY_double);
  d_filterRhoWLabel   =  VarLabel::create("filterRhoW",   SFCZ_double);
  d_filterRhoLabel    =  VarLabel::create("filterRho",    CC_double );
  d_filterRhoFLabel   =  VarLabel::create("filterRhoF",   CC_double );
  d_filterRhoELabel   =  VarLabel::create("filterRhoE",   CC_double );
  d_filterRhoRFLabel  =  VarLabel::create("filterRhoRF",  CC_double );
 
  d_scalarGradientCompLabel             =  VarLabel::create("scalarGradientComp",             CC_double);
  d_filterScalarGradientCompLabel       =  VarLabel::create("filterScalarGradientComp",       CC_double);
  d_enthalpyGradientCompLabel           =  VarLabel::create("enthalpyGradientComp",           CC_double);
  d_filterEnthalpyGradientCompLabel     =  VarLabel::create("filterEnthalpyGradientComp",     CC_double);
  d_reactScalarGradientCompLabel        =  VarLabel::create("reactScalarGradientComp",        CC_double);
  d_filterReactScalarGradientCompLabel  =  VarLabel::create("filterReactScalarGradientComp",  CC_double);
  d_filterStrainTensorCompLabel         =  VarLabel::create("filterStrainTensorComp",         CC_double);
  d_filterVolumeLabel                   =  VarLabel::create("filterVolume",                   CC_double);

  d_scalarNumeratorLabel         =  VarLabel::create("scalarNumerator",         CC_double);
  d_scalarDenominatorLabel       =  VarLabel::create("scalarDenominator",       CC_double);
  d_enthalpyNumeratorLabel       =  VarLabel::create("enthalpyNumerator",       CC_double);
  d_enthalpyDenominatorLabel     =  VarLabel::create("enthalpyDenominator",     CC_double);
  d_reactScalarNumeratorLabel    =  VarLabel::create("reactScalarNumerator",    CC_double);
  d_reactScalarDenominatorLabel  =  VarLabel::create("reactScalarDenominator",  CC_double);

  d_ShFLabel   =  VarLabel::create("ShF",   CC_double);
  d_ShELabel   =  VarLabel::create("ShE",   CC_double);
  d_ShRFLabel  =  VarLabel::create("ShRF",  CC_double);

  
  //MMS labels
  d_uFmmsLabel  =  VarLabel::create("uFmms",  SFCX_double);
  d_vFmmsLabel  =  VarLabel::create("vFmms",  SFCY_double);
  d_wFmmsLabel  =  VarLabel::create("wFmms",  SFCZ_double);

  //A helper variable 
  d_zerosrcVarLabel = VarLabel::create("zerosrcVar",  CC_double);

  //rate Labels ~ may require a more elegant solution later?
  d_co2RateLabel = VarLabel::create("co2Rate",  CC_double);
  d_so2RateLabel = VarLabel::create("so2Rate",  CC_double);

  //Artificial source terms
  d_scalarBoundarySrcLabel    =  VarLabel::create("scalarBoundarySrc",    CC_double);
  d_enthalpyBoundarySrcLabel  =  VarLabel::create("enthalpyBoundarySrc",  CC_double);
  d_umomBoundarySrcLabel      =  VarLabel::create("umomBoundarySrc",      SFCX_double);
  d_vmomBoundarySrcLabel      =  VarLabel::create("vmomBoundarySrc",      SFCY_double);
  d_wmomBoundarySrcLabel      =  VarLabel::create("wmomBoundarySrc",      SFCZ_double);

  //DQMOM vars
  
  // Boundary condition variables
  d_areaFractionLabel         = VarLabel::create("areaFraction", CC_Vector); 
  d_volFractionLabel          = VarLabel::create("volFraction", CC_double); 
  d_areaFractionFXLabel       = VarLabel::create("areaFractionFX", SFCX_double);
  d_areaFractionFYLabel       = VarLabel::create("areaFractionFY", SFCY_double);
  d_areaFractionFZLabel       = VarLabel::create("areaFractionFZ", SFCZ_double);

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
  VarLabel::destroy(d_cellInfoLabel);
  VarLabel::destroy(d_cellTypeLabel);
  VarLabel::destroy(d_totalflowINLabel);
  VarLabel::destroy(d_totalflowOUTLabel);
  VarLabel::destroy(d_netflowOUTBCLabel);
  VarLabel::destroy(d_totalAreaOUTLabel);
  VarLabel::destroy(d_denAccumLabel);
  VarLabel::destroy(d_densityCPLabel);
  VarLabel::destroy(d_densityGuessLabel);
  VarLabel::destroy(d_densityTempLabel);
  VarLabel::destroy(d_densityOldOldLabel);
  VarLabel::destroy(d_filterdrhodtLabel);
  VarLabel::destroy(d_drhodfCPLabel);
  VarLabel::destroy(d_viscosityCTSLabel);
  VarLabel::destroy(d_tauSGSLabel); 
  VarLabel::destroy(d_scalarDiffusivityLabel);
  VarLabel::destroy(d_enthalpyDiffusivityLabel);
  VarLabel::destroy(d_reactScalarDiffusivityLabel);
  VarLabel::destroy(d_pressurePSLabel);
  VarLabel::destroy(d_pressureGuessLabel);
  VarLabel::destroy(d_pressureExtraProjectionLabel);
  VarLabel::destroy(d_presCoefPBLMLabel);
  VarLabel::destroy(d_presNonLinSrcPBLMLabel);
  VarLabel::destroy(d_uVelocitySPBCLabel);
  VarLabel::destroy(d_vVelocitySPBCLabel);
  VarLabel::destroy(d_wVelocitySPBCLabel);
  VarLabel::destroy(d_scalarSPLabel);
  VarLabel::destroy(d_scalarTempLabel);
  VarLabel::destroy(d_scalarFELabel);
  VarLabel::destroy(d_scalarVarSPLabel);
  VarLabel::destroy(d_scalarDissSPLabel);
  VarLabel::destroy(d_scalCoefSBLMLabel);
  VarLabel::destroy(d_scalarTotCoefLabel);
  VarLabel::destroy(d_scalNonLinSrcSBLMLabel);
  VarLabel::destroy(d_reactscalarSPLabel);
  VarLabel::destroy(d_reactscalarTempLabel);
  VarLabel::destroy(d_reactscalarFELabel);
  VarLabel::destroy(d_reactscalCoefSBLMLabel);
  VarLabel::destroy(d_reactscalNonLinSrcSBLMLabel);
  VarLabel::destroy(d_refDensity_label);
  VarLabel::destroy(d_refDensityPred_label);
  VarLabel::destroy(d_refPressurePred_label);
  VarLabel::destroy(d_refPressure_label);
  VarLabel::destroy(d_CCVelocityLabel);
  VarLabel::destroy(d_mmcellTypeLabel);
  VarLabel::destroy(d_mmgasVolFracLabel);

  VarLabel::destroy(d_dummyTLabel);
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

  VarLabel::destroy(d_s2INLabel);
  VarLabel::destroy(d_shINLabel);
  VarLabel::destroy(d_soINLabel);
  VarLabel::destroy(d_hso2INLabel);

  VarLabel::destroy(d_hosoINLabel);
  VarLabel::destroy(d_hoso2INLabel);
  VarLabel::destroy(d_snINLabel);
  VarLabel::destroy(d_csINLabel);

  VarLabel::destroy(d_ocsINLabel);
  VarLabel::destroy(d_hsoINLabel);
  VarLabel::destroy(d_hosINLabel);
  VarLabel::destroy(d_hsohINLabel);

  VarLabel::destroy(d_h2soINLabel);
  VarLabel::destroy(d_hoshoINLabel);
  VarLabel::destroy(d_hs2INLabel);
  VarLabel::destroy(d_h2s2INLabel);

  VarLabel::destroy(d_coINLabel);
  VarLabel::destroy(d_c2h2INLabel);
  VarLabel::destroy(d_ch4INLabel);
  VarLabel::destroy(d_mixMWLabel); 
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
  VarLabel::destroy(d_enthalpyTempLabel);
  VarLabel::destroy(d_enthalpyFELabel);
  VarLabel::destroy(d_enthalpyRXNLabel);
  VarLabel::destroy(d_enthCoefSBLMLabel);
  VarLabel::destroy(d_enthNonLinSrcSBLMLabel);
  VarLabel::destroy(d_absorpINLabel);
  VarLabel::destroy(d_abskgINLabel);
  VarLabel::destroy(d_sootFVINLabel);
  VarLabel::destroy(d_radiationSRCINLabel);
  VarLabel::destroy(d_radiationFluxEINLabel);
  VarLabel::destroy(d_radiationFluxWINLabel);
  VarLabel::destroy(d_radiationFluxNINLabel);
  VarLabel::destroy(d_radiationFluxSINLabel);
  VarLabel::destroy(d_radiationFluxTINLabel);
  VarLabel::destroy(d_radiationFluxBINLabel);
  VarLabel::destroy(d_radiationVolqINLabel);
  VarLabel::destroy(d_reactscalarSRCINLabel);
  
 // Runge-Kutta 3d order properties labels
  VarLabel::destroy(d_refDensityInterm_label);
  VarLabel::destroy(d_refPressureInterm_label);
 // Runge-Kutta 3d order pressure and momentum labels
  VarLabel::destroy(d_pressureIntermLabel);
// labels for scale similarity model
  VarLabel::destroy(d_stressTensorCompLabel); 
  VarLabel::destroy(d_strainTensorCompLabel);
  VarLabel::destroy(d_LIJCompLabel);
  VarLabel::destroy(d_scalarFluxCompLabel);
  VarLabel::destroy(d_velocityDivergenceLabel);
  VarLabel::destroy(d_vorticityXLabel);
  VarLabel::destroy(d_vorticityYLabel);
  VarLabel::destroy(d_vorticityZLabel);
  VarLabel::destroy(d_vorticityLabel);
  VarLabel::destroy(d_velDivResidualLabel);
  VarLabel::destroy(d_continuityResidualLabel);

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
  VarLabel::destroy(d_filterVolumeLabel); 
  VarLabel::destroy(d_scalarNumeratorLabel); 
  VarLabel::destroy(d_scalarDenominatorLabel); 
  VarLabel::destroy(d_enthalpyNumeratorLabel); 
  VarLabel::destroy(d_enthalpyDenominatorLabel); 
  VarLabel::destroy(d_reactScalarNumeratorLabel); 
  VarLabel::destroy(d_reactScalarDenominatorLabel); 
  VarLabel::destroy(d_ShFLabel);
  VarLabel::destroy(d_ShELabel);
  VarLabel::destroy(d_ShRFLabel);

  //mms variabels
  VarLabel::destroy(d_uFmmsLabel);
  VarLabel::destroy(d_vFmmsLabel);
  VarLabel::destroy(d_wFmmsLabel);

  VarLabel::destroy(d_zerosrcVarLabel);

  VarLabel::destroy(d_co2RateLabel);
  VarLabel::destroy(d_so2RateLabel);

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

  VarLabel::destroy(d_scalarBoundarySrcLabel);
  VarLabel::destroy(d_enthalpyBoundarySrcLabel);
  VarLabel::destroy(d_umomBoundarySrcLabel);
  VarLabel::destroy(d_vmomBoundarySrcLabel);
  VarLabel::destroy(d_wmomBoundarySrcLabel);
  
  VarLabel::destroy(d_tempFxLabel);
  VarLabel::destroy(d_tempFyLabel);
  VarLabel::destroy(d_tempFzLabel);

  VarLabel::destroy(d_areaFractionLabel); 
  VarLabel::destroy(d_areaFractionFXLabel); 
  VarLabel::destroy(d_areaFractionFYLabel); 
  VarLabel::destroy(d_areaFractionFZLabel); 
  VarLabel::destroy(d_volFractionLabel); 

  for (PartVelMap::iterator i = partVel.begin(); i != partVel.end(); i++){
    VarLabel::destroy(i->second); 
  }
  
}           

void ArchesLabel::setSharedState(SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
}

void ArchesLabel::problemSetup( const ProblemSpecP& db )
{ 

  ProblemSpecP db_lab = db->findBlock("VarID"); 

  if ( db_lab ){ 

    for ( ProblemSpecP d = db_lab->findBlock("var"); d != 0; d = d->findNextBlock("var") ){ 

      std::string label; 
      std::string role; 
      d->getAttribute("role", role);
      d->getAttribute("label",label); 

      setVarlabelToRole( label, role ); 

    } 
  } 
} 
/** @brief Retrieve a label based on its CFD role **/
const VarLabel* ArchesLabel::getVarlabelByRole( const std::string role ){ 

   RLMAP::iterator i = d_r_to_l.find(role);
   const VarLabel* the_label; 

   if ( i != d_r_to_l.end() ){

     the_label = VarLabel::find(i->second); 

     if ( the_label != NULL ){

      return the_label; 

     } else { 

      std::string msg = "Error: Label not recognized in <VarID> storage for role = "+role+"\n";;
      throw InvalidValue(msg,__FILE__,__LINE__);

     } 

   } else { 

     std::string msg = "Error: Role not found in <VarID> storage for role = "+role+"\n";
     throw InvalidValue(msg,__FILE__,__LINE__);

   } 
}; 

/** @brief Set a label to have a specific role **/ 
void ArchesLabel::setVarlabelToRole( const std::string label, const std::string role ){ 

  //first make sure that the role is allowed: 
  //allowed roles are defined in the constructor of this class. 
  bool found_role = false; 
  for ( std::vector<std::string>::iterator i_r = d_allowed_roles.begin(); i_r != d_allowed_roles.end(); i_r++){ 

    if ( *i_r == role ){ 

      found_role = true; 

    } 
  } 
  if ( !found_role ){ 

    std::string msg = "Error: Trying to assign a role "+role+" which is not defined as valid in ArchesLabel.cc\n";
    throw InvalidValue(msg,__FILE__,__LINE__); 

  } 


  RLMAP::iterator i = d_r_to_l.find( role ); 

  if ( i != d_r_to_l.end() ){ 
    //if this role and found, make sure that the label is consistent with 
    //what you are trying to load: 
    if ( label != i->second ){ 
      std::string msg = "Error: Trying to specify "+label+" for role "+role+" which already has the label"+i->second+"\n";
      throw InvalidValue(msg, __FILE__,__LINE__); 
    } // else the variable is already identified with its role so no need to do anything...
  } else { 
    //not found...so insert it
    d_r_to_l.insert(std::make_pair(role,label)); 
  } 

}; 
