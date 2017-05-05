/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

   // shortcuts
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  const TypeDescription* CC_Vector = CCVariable<Vector>::getTypeDescription();

  const TypeDescription* SFCX_double = SFCXVariable<double>::getTypeDescription();
  const TypeDescription* SFCY_double = SFCYVariable<double>::getTypeDescription();
  const TypeDescription* SFCZ_double = SFCZVariable<double>::getTypeDescription();

  const TypeDescription* sum_variable = sum_vartype::getTypeDescription();

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
  // Density Labels
  d_densityCPLabel      =  VarLabel::create("densityCP",      CC_double);
  d_densityGuessLabel   =  VarLabel::create("densityGuess",   CC_double);
  d_densityTempLabel    =  VarLabel::create("densityTemp",    CC_double);
  d_filterdrhodtLabel   =  VarLabel::create("filterdrhodt",   CC_double);

  // Viscosity Labels
  d_viscosityCTSLabel           = VarLabel::create("viscosityCTS",            CC_double);
  d_turbViscosLabel             = VarLabel::create("turb_viscosity",          CC_double);

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

  d_uMomLabel = VarLabel::create("Umom", SFCX_double);
  d_vMomLabel = VarLabel::create("Vmom", SFCY_double);
  d_wMomLabel = VarLabel::create("Wmom", SFCZ_double);
  d_conv_scheme_x_Label = VarLabel::create("conv_scheme_x", SFCX_double);
  d_conv_scheme_y_Label = VarLabel::create("conv_scheme_y", SFCY_double);
  d_conv_scheme_z_Label = VarLabel::create("conv_scheme_z", SFCZ_double);

  // labels for ref density and pressure
  d_refDensity_label      =  VarLabel::create("refDensityLabel",      sum_variable); //ref density only used in compdynamicproc
  d_refDensityPred_label  =  VarLabel::create("refDensityPredLabel",  sum_variable);
  d_refPressurePred_label =  VarLabel::create("refPressurePredLabel", sum_variable); //ref pressure only used in the pressure solver
  d_refPressure_label     =  VarLabel::create("refPressureLabel",     sum_variable);

  // Cell Centered data after interpolation (for use in visualization and turbulence models)
  d_CCVelocityLabel   =  VarLabel::create("CCVelocity",      CC_Vector);
  d_CCUVelocityLabel  =  VarLabel::create("CCUVelocity",     CC_double);
  d_CCVVelocityLabel  =  VarLabel::create("CCVVelocity",     CC_double);
  d_CCWVelocityLabel  =  VarLabel::create("CCWVelocity",     CC_double);

  // multimaterial wall/intrusion cells
  d_mmcellTypeLabel = VarLabel::create("mmcellType",
                                      CCVariable<int>::getTypeDescription());

  // Label for void fraction, after correction for wall cells using cutoff
  d_mmgasVolFracLabel = VarLabel::create("mmgasVolFrac",  CC_double);


  // Microscopic density (i.e., without void fraction) of gas
  d_densityMicroLabel    =  VarLabel::create("denMicro",    CC_double);
  d_densityMicroINLabel  =  VarLabel::create("denMicroIN",  CC_double);

  // Sum of the relative pressure and the hydrostatic contribution
  d_pressPlusHydroLabel = VarLabel::create("pPlusHydro",  CC_double);

  // for corrector step
  d_uVelRhoHatLabel     =  VarLabel::create("uvelRhoHat",     SFCX_double);
  d_vVelRhoHatLabel     =  VarLabel::create("vvelRhoHat",     SFCY_double);
  d_wVelRhoHatLabel     =  VarLabel::create("wvelRhoHat",     SFCZ_double);

  d_uVelRhoHat_CCLabel  =  VarLabel::create("uvelRhoHat_CC",  CC_double );
  d_vVelRhoHat_CCLabel  =  VarLabel::create("vvelRhoHat_CC",  CC_double );
  d_wVelRhoHat_CCLabel  =  VarLabel::create("wvelRhoHat_CC",  CC_double );

  d_pressurePredLabel = VarLabel::create("pressurePred",    CC_double);

  //__________________________________
  // Radiation
  d_absorpINLabel   =  VarLabel::create("absorpIN",   CC_double);
  d_abskgINLabel    =  VarLabel::create("abskgIN",    CC_double);

  d_radiationSRCINLabel = VarLabel::create("radiationSRCIN",  CC_double);

  d_radiationFluxEINLabel = VarLabel::create("radiationFluxEIN",  CC_double);
  d_radiationFluxWINLabel = VarLabel::create("radiationFluxWIN",  CC_double);
  d_radiationFluxNINLabel = VarLabel::create("radiationFluxNIN",  CC_double);
  d_radiationFluxSINLabel = VarLabel::create("radiationFluxSIN",  CC_double);
  d_radiationFluxTINLabel = VarLabel::create("radiationFluxTIN",  CC_double);
  d_radiationFluxBINLabel = VarLabel::create("radiationFluxBIN",  CC_double);
  d_radiationVolqINLabel = VarLabel::create("radiationVolqIN",  CC_double);

  //__________________________________
  // Scalesimilarity
  d_stressTensorCompLabel  =  VarLabel::create("stressTensorComp",  CC_double );

  d_strainTensorCompLabel  =  VarLabel::create("strainTensorComp",  CC_double);
  d_LIJCompLabel           =  VarLabel::create("LIJComp",           CC_double);

  //__________________________________
  // Dynamic procedure
  d_strainMagnitudeLabel    =  VarLabel::create("strainMagnitudeLabel",    CC_double);
  d_strainMagnitudeMLLabel  =  VarLabel::create("strainMagnitudeMLLabel",  CC_double);
  d_strainMagnitudeMMLabel  =  VarLabel::create("strainMagnitudeMMLabel",  CC_double);
  d_LalphaLabel             =  VarLabel::create("lalphaLabel",             CC_double);
  d_cbetaHATalphaLabel      =  VarLabel::create("cbetaHATalphaLabel",      CC_double);
  d_alphaalphaLabel         =  VarLabel::create("alphaalphaLabel",         CC_double);
  d_CsLabel                 =  VarLabel::create("CsLabel",                 CC_double);

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

  d_negativeDensityGuess_label           =  VarLabel::create("negativeDensityGuess",           sum_variable); // Used in EnthalpySolver and ExplicitSolver (also computed here)
  d_negativeDensityGuessPred_label       =  VarLabel::create("negativeDensityGuessPred",       sum_variable);
  d_negativeDensityGuessInterm_label     =  VarLabel::create("negativeDensityGuessInterm",     sum_variable);
  d_densityLag_label                     =  VarLabel::create("densityLag",                     sum_variable); // Only computed/used if maxDensityLag is used in the input file.
  d_densityLagPred_label                 =  VarLabel::create("densityLagPred",                 sum_variable); // will restart the timestep if (guess_rho - computed_rho) > lag_rho
  d_densityLagInterm_label               =  VarLabel::create("densityLagInterm",               sum_variable);
  d_densityLagAfterAverage_label         =  VarLabel::create("densityLagAfterAverage",         sum_variable);
  d_densityLagAfterIntermAverage_label   =  VarLabel::create("densityLagAfterIntermAverage",   sum_variable);

// kinetic energy
  d_kineticEnergyLabel             =  VarLabel::create("kineticEnergy",             CC_double);
  d_totalKineticEnergyLabel        =  VarLabel::create("totalKineticEnergy",        sum_variable); //only computes if turned on from input file

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

  d_filterScalarGradientCompLabel       =  VarLabel::create("filterScalarGradientComp",       CC_double);
  d_filterEnthalpyGradientCompLabel     =  VarLabel::create("filterEnthalpyGradientComp",     CC_double);
  d_filterReactScalarGradientCompLabel  =  VarLabel::create("filterReactScalarGradientComp",  CC_double);
  d_filterStrainTensorCompLabel         =  VarLabel::create("filterStrainTensorComp",         CC_double);
  d_filterVolumeLabel                   =  VarLabel::create("filterVolume",                   CC_double);

  d_ShFLabel   =  VarLabel::create("ShF",   CC_double);
  d_ShELabel   =  VarLabel::create("ShE",   CC_double);
  d_ShRFLabel  =  VarLabel::create("ShRF",  CC_double);

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
  if (d_stencilMatl->removeReference()) {
    delete d_stencilMatl;
  }

  if (d_vectorMatl->removeReference()) {
    delete d_vectorMatl;
  }

  if (d_tensorMatl->removeReference()) {
    delete d_tensorMatl;
  }

  if (d_symTensorMatl->removeReference()) {
    delete d_symTensorMatl;
  }
  for( ArchesLabel::MomentMap::iterator iMoment = DQMOMMoments.begin(); iMoment != DQMOMMoments.end(); ++iMoment ) {
    VarLabel::destroy(iMoment->second); 
  }

  // Get rid of cqmom var labels
  for( ArchesLabel::WeightMap::iterator iW = CQMOMWeights.begin(); iW != CQMOMWeights.end(); ++iW ) {
    VarLabel::destroy(iW->second);
  }

  for( ArchesLabel::AbscissaMap::iterator iA = CQMOMAbscissas.begin(); iA != CQMOMAbscissas.end(); ++iA ) {
    VarLabel::destroy(iA->second);
  }

  VarLabel::destroy(d_strainMagnitudeLabel);
  VarLabel::destroy(d_strainMagnitudeMLLabel);
  VarLabel::destroy(d_strainMagnitudeMMLabel);
  VarLabel::destroy(d_LalphaLabel);
  VarLabel::destroy(d_cbetaHATalphaLabel);
  VarLabel::destroy(d_alphaalphaLabel);
  VarLabel::destroy(d_CsLabel);
  VarLabel::destroy(d_cellInfoLabel);
  VarLabel::destroy(d_cellTypeLabel);
  VarLabel::destroy(d_densityCPLabel);
  VarLabel::destroy(d_densityGuessLabel);
  VarLabel::destroy(d_densityTempLabel);
  VarLabel::destroy(d_filterdrhodtLabel);
  VarLabel::destroy(d_viscosityCTSLabel);
  VarLabel::destroy(d_turbViscosLabel);
  VarLabel::destroy(d_pressurePSLabel);
  VarLabel::destroy(d_pressureGuessLabel);
  VarLabel::destroy(d_pressureExtraProjectionLabel);
  VarLabel::destroy(d_presCoefPBLMLabel);
  VarLabel::destroy(d_presNonLinSrcPBLMLabel);
  VarLabel::destroy(d_uVelocitySPBCLabel);
  VarLabel::destroy(d_vVelocitySPBCLabel);
  VarLabel::destroy(d_wVelocitySPBCLabel);
  VarLabel::destroy(d_uMomLabel);
  VarLabel::destroy(d_vMomLabel);
  VarLabel::destroy(d_wMomLabel);
  VarLabel::destroy(d_conv_scheme_x_Label);
  VarLabel::destroy(d_conv_scheme_y_Label);
  VarLabel::destroy(d_conv_scheme_z_Label);
  VarLabel::destroy(d_refDensity_label);
  VarLabel::destroy(d_refDensityPred_label);
  VarLabel::destroy(d_refPressurePred_label);
  VarLabel::destroy(d_refPressure_label);
  VarLabel::destroy(d_CCVelocityLabel);
  VarLabel::destroy(d_CCUVelocityLabel);
  VarLabel::destroy(d_CCVVelocityLabel);
  VarLabel::destroy(d_CCWVelocityLabel);

  VarLabel::destroy(d_mmcellTypeLabel);
  VarLabel::destroy(d_mmgasVolFracLabel);

  VarLabel::destroy(d_densityMicroLabel);
  VarLabel::destroy(d_densityMicroINLabel);
  VarLabel::destroy(d_pressPlusHydroLabel);
  VarLabel::destroy(d_uVelRhoHatLabel);
  VarLabel::destroy(d_vVelRhoHatLabel);
  VarLabel::destroy(d_wVelRhoHatLabel);
  VarLabel::destroy(d_uVelRhoHat_CCLabel);
  VarLabel::destroy(d_vVelRhoHat_CCLabel);
  VarLabel::destroy(d_wVelRhoHat_CCLabel);
  VarLabel::destroy(d_pressurePredLabel);
  VarLabel::destroy(d_absorpINLabel);
  VarLabel::destroy(d_abskgINLabel);
  VarLabel::destroy(d_radiationSRCINLabel);
  VarLabel::destroy(d_radiationFluxEINLabel);
  VarLabel::destroy(d_radiationFluxWINLabel);
  VarLabel::destroy(d_radiationFluxNINLabel);
  VarLabel::destroy(d_radiationFluxSINLabel);
  VarLabel::destroy(d_radiationFluxTINLabel);
  VarLabel::destroy(d_radiationFluxBINLabel);
  VarLabel::destroy(d_radiationVolqINLabel);

 // Runge-Kutta 3d order properties labels
  VarLabel::destroy(d_refDensityInterm_label);
  VarLabel::destroy(d_refPressureInterm_label);
 // Runge-Kutta 3d order pressure and momentum labels
  VarLabel::destroy(d_pressureIntermLabel);
// labels for scale similarity model
  VarLabel::destroy(d_stressTensorCompLabel);
  VarLabel::destroy(d_strainTensorCompLabel);
  VarLabel::destroy(d_LIJCompLabel);
  VarLabel::destroy(d_velocityDivergenceLabel);
  VarLabel::destroy(d_vorticityXLabel);
  VarLabel::destroy(d_vorticityYLabel);
  VarLabel::destroy(d_vorticityZLabel);
  VarLabel::destroy(d_vorticityLabel);
  VarLabel::destroy(d_velDivResidualLabel);
  VarLabel::destroy(d_continuityResidualLabel);

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
  VarLabel::destroy(d_oldDeltaTLabel);

// test filtered terms for variable density dynamic Smagorinsky model
  VarLabel::destroy(d_filterRhoULabel);
  VarLabel::destroy(d_filterRhoVLabel);
  VarLabel::destroy(d_filterRhoWLabel);
  VarLabel::destroy(d_filterRhoLabel);
  VarLabel::destroy(d_filterRhoFLabel);
  VarLabel::destroy(d_filterRhoELabel);
  VarLabel::destroy(d_filterRhoRFLabel);
  VarLabel::destroy(d_filterScalarGradientCompLabel);
  VarLabel::destroy(d_filterEnthalpyGradientCompLabel);
  VarLabel::destroy(d_filterReactScalarGradientCompLabel);
  VarLabel::destroy(d_filterStrainTensorCompLabel);
  VarLabel::destroy(d_filterVolumeLabel);
  VarLabel::destroy(d_ShFLabel);
  VarLabel::destroy(d_ShELabel);
  VarLabel::destroy(d_ShRFLabel);

  VarLabel::destroy(d_areaFractionLabel);
  VarLabel::destroy(d_areaFractionFXLabel);
  VarLabel::destroy(d_areaFractionFYLabel);
  VarLabel::destroy(d_areaFractionFZLabel);
  VarLabel::destroy(d_volFractionLabel);

  for (PartVelMap::iterator i = partVel.begin(); i != partVel.end(); i++){
    VarLabel::destroy(i->second);
  }
}

void
ArchesLabel::setSharedState(SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
}

void
ArchesLabel::problemSetup( const ProblemSpecP& db )
{
  ProblemSpecP db_lab = db->findBlock("VarID");

  if ( db_lab ){

    for ( ProblemSpecP d = db_lab->findBlock("var"); d != nullptr; d = d->findNextBlock("var") ){ 

      std::string label;
      std::string role;
      d->getAttribute("role", role);
      d->getAttribute("label",label);

      setVarlabelToRole( label, role );
    }
  }
}
/** @brief Retrieve a label based on its CFD role **/
const VarLabel *
ArchesLabel::getVarlabelByRole( VARID role )
{
   RLMAP::iterator i = d_r_to_l.find(role);
   const VarLabel* the_label;

   const std::string role_name = getRoleString( role );

   if ( i != d_r_to_l.end() ){

     the_label = VarLabel::find(i->second);

     if ( the_label != nullptr ){

      return the_label;

     } else {

      std::string msg = "Error: Label not recognized in <VarID> storage for role = "+role_name+"\n";;
      throw InvalidValue(msg,__FILE__,__LINE__);

     }

   } else {

     std::string msg = "Error: Role not found in <VarID> storage for role = "+role_name+"\n";
     throw InvalidValue(msg,__FILE__,__LINE__);

   }
};

/** @brief Set a label to have a specific role **/ 
void
ArchesLabel::setVarlabelToRole( const std::string label, const std::string role ){ 

  // First make sure that the role is allowed:
  // Allowed roles are defined in the constructor of this class.
  bool found_role = true;
  VARID which_var;
  if ( role == "temperature" ) {
    which_var = TEMPERATURE;
  }
  else if ( role == "density" ) {
    which_var = DENSITY;
  }
  else if ( role == "enthalpy" ) {
    which_var = ENTHALPY;
  }
  else if ( role == "co2" ) {
    which_var = CO2;
  }
  else if ( role == "h2o" ) {
    which_var = H2O;
  }
  else if ( role == "co2" ) {
    which_var = CO2;
  }
  else if ( role == "soot" ) {
    which_var = SOOT;
  }
  else {
    found_role = false;
  }

  if ( !found_role ){

    std::string msg = "Error: Trying to assign a role "+role+" which is not defined as valid in ArchesLabel.cc\n";
    throw InvalidValue(msg,__FILE__,__LINE__);

  }

  RLMAP::iterator i = d_r_to_l.find( which_var );

  if ( i != d_r_to_l.end() ){
    //if this role and found, make sure that the label is consistent with
    //what you are trying to load:
    if ( label != i->second ){
      std::string msg = "Error: Trying to specify "+label+" for role "+role+" which already has the label"+i->second+"\n";
      throw InvalidValue(msg, __FILE__,__LINE__);
    } // else the variable is already identified with its role so no need to do anything...
  }
  else {
    //not found...so insert it
    d_r_to_l.insert(std::make_pair(which_var,label));
  }

};

const string
ArchesLabel::getRoleString( VARID role )
{
  std::string name; 
  if ( role == TEMPERATURE ){ 
    name = "temperature"; 
  }
  else if ( role == DENSITY ){ 
    name = "density"; 
  }
  else if ( role == ENTHALPY ){ 
    name = "enthalpy"; 
  }
  else if ( role == SOOT ){ 
    name = "soot"; 
  }
  else if ( role == CO2 ){ 
    name = "co2"; 
  }
  else if ( role == H2O ){ 
    name = "h2o"; 
  }
  else if ( role == SPECIFICHEAT ){ 
    name = "specific_heat"; 
  }
  else if ( role == MIXTUREFRACTION ){ 
    name = "mixture_fraction"; 
  }
  else { 
    std::string msg = "Error: Role not recognized!\n";
    throw InvalidValue( msg, __FILE__, __LINE__ );
  }
  return name;
}
