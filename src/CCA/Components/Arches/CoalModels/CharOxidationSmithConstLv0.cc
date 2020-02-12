/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Arches/CoalModels/CharOxidationSmithConstLv0.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <sci_defs/visit_defs.h>

#include <iomanip>
#include <iostream>
#include <numeric>

#define SQUARE(x) x*x
#define CUBE(x) x*x*x
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
CharOxidationSmithConstLv0Builder::CharOxidationSmithConstLv0Builder( const std::string         & modelName,
                                                      const std::vector<std::string> & reqICLabelNames,
                                                      const std::vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      MaterialManagerP          & materialManager,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
}

CharOxidationSmithConstLv0Builder::~CharOxidationSmithConstLv0Builder(){}

ModelBase* CharOxidationSmithConstLv0Builder::build() {
  return scinew CharOxidationSmithConstLv0( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationSmithConstLv0::CharOxidationSmithConstLv0( std::string modelName,
                                        MaterialManagerP& materialManager,
                                        ArchesLabel* fieldLabels,
                                        std::vector<std::string> icLabelNames,
                                        std::vector<std::string> scalarLabelNames,
                                        int qn )
: CharOxidation(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Set constants
  // Enthalpy of formation (J/mol)
  _HF_CO2 = -393509.0;
  _HF_CO  = -110525.0;
  //binary diffsuion at 293 K
  _T0 = 293.0;
  // ideal gas constants
  _R_cal = 1.9872036; // [cal/ (K mol) ]
  _R = 8.314; // [J/ (K mol) ]

  _char_birth_label = nullptr;
  _rawcoal_birth_label = nullptr;

}

CharOxidationSmithConstLv0::~CharOxidationSmithConstLv0()
{
  for (std::vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    VarLabel::destroy( *iter );
  }
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CharOxidationSmithConstLv0::problemSetup(const ProblemSpecP& params, int qn)
{
  // Ensure the following species are populated from table
  // (this is expensive and should be avoided, if a species isn't needed)
  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species("temperature");
  helper.add_lookup_species("mixture_molecular_weight");

  // Example on getting the table constants
   ChemHelper::TableConstantsMapType the_table_constants = helper.get_table_constants();
   auto press_iter = the_table_constants->find("Pressure");
   if ( press_iter == the_table_constants->end() ){
     _gasPressure=101325.;
     if (qn==0)
       proc0cout << " No Pressure key found in the table." << std::endl;
   }else{
     _gasPressure=press_iter->second;
     if (qn==0)
       proc0cout << " Pressure used in char OXY smith." <<_gasPressure << " pascals" << std::endl;
   }


  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    std::string particleType;
    db_coal_props->getAttribute("type",particleType);
    if (particleType != "coal"){
      throw InvalidValue("ERROR: CharOxidationSmithConstLv0: Can't use particles of type: "+particleType,__FILE__,__LINE__);
    }

  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", _dynamic_visc);
  } else {
    throw InvalidValue("Error: Missing <PhysicalConstants> section in input file required for Smith Char Oxidation model.",__FILE__,__LINE__);
  }

  // create raw coal mass var label
  std::string rcmass_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
  std::string rcmass_name = ArchesCore::append_env( rcmass_root, d_quadNode );
  std::string rcmassqn_name = ArchesCore::append_qn_env(rcmass_root, d_quadNode );
  _rcmass_varlabel = VarLabel::find(rcmass_name);

  // check for char mass and get scaling constant
  std::string char_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
  std::string char_name = ArchesCore::append_env( char_root, d_quadNode );
  std::string charqn_name = ArchesCore::append_qn_env( char_root, d_quadNode );
  _char_varlabel = VarLabel::find(char_name);

  EqnBase& temp_char_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(charqn_name);
  DQMOMEqn& char_eqn = dynamic_cast<DQMOMEqn&>(temp_char_eqn);
  _char_scaling_constant = char_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = charqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  //CHAR get the birth term if any:
  const std::string char_birth_name = char_eqn.get_model_by_type( "BirthDeath" );
  std::string char_birth_qn_name = ArchesCore::append_qn_env(char_birth_name, d_quadNode);
  if ( char_birth_name != "NULLSTRING" ){
    _char_birth_label = VarLabel::find( char_birth_qn_name );
  }

  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
  _RC_scaling_constant  = rcmass_eqn.getScalingConstant(d_quadNode)  ;
  std::string RC_RHS = rcmassqn_name + "_RHS";
  _RC_RHS_source_varlabel = VarLabel::find(RC_RHS);

  //RAW COAL get the birth term if any:
  const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
  std::string rawcoal_birth_qn_name = ArchesCore::append_qn_env(rawcoal_birth_name, d_quadNode);
  if ( rawcoal_birth_name != "NULLSTRING" ){
    _rawcoal_birth_label = VarLabel::find( rawcoal_birth_qn_name );
  }

  // check for particle temperature
  std::string temperature_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);
  std::string temperature_name = ArchesCore::append_env( temperature_root, d_quadNode );
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
  if(_particle_temperature_varlabel == 0){
    throw ProblemSetupException("Error: Unable to find coal temperature label!!!! Looking for name: "+temperature_name, __FILE__, __LINE__);
  }

  // check for length
  _nQn_part = ArchesCore::get_num_env(db,ArchesCore::DQMOM_METHOD);
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  for (int i=0; i<_nQn_part;i++ ){
    std::string length_name = ArchesCore::append_env( length_root, i );
    _length_varlabel.push_back(  VarLabel::find(length_name));
  }

  // get weight scaling constant
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode);
  for (int i=0; i<_nQn_part;i++ ){
  std::string weight_name = ArchesCore::append_env("w", i);
    _weight_varlabel.push_back( VarLabel::find(weight_name) );
  }
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

  std::string number_density_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TOTNUM_DENSITY);
  _number_density_varlabel = VarLabel::find(number_density_name);

  // get Char source term label and devol label from the devolatilization model
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      _devolRCLabel = iModel->second->getModelLabel() ;
    }
  }


  // model global constants
  // get model coefficients
  std::string oxidizer_name;
  double oxidizer_MW; //
  double hrxn; //
  bool use_co2co;
  double a; //
  double e; //
  double phi; //
  if (db_coal_props->findBlock("SmithCharConstLv0")) {
    ProblemSpecP db_Smith = db_coal_props->findBlock("SmithCharConstLv0");
    db_Smith->getWithDefault("char_MW",_Mh,12.0); // kg char / kmole char
    db_Smith->getWithDefault("surface_area_mult_factor",_S,1.0);
    _NUM_species = 0;
    for ( ProblemSpecP db_species = db_Smith->findBlock( "species" ); db_species != nullptr; db_species = db_species->findNextBlock( "species" ) ){
      std::string new_species;
      new_species = db_species->getNodeValue();
      helper.add_lookup_species(new_species);
      _species_names.push_back(new_species);
      _NUM_species += 1;
    }
    _NUM_reactions = 0;
    for ( ProblemSpecP db_reaction = db_Smith->findBlock( "reaction" ); db_reaction != nullptr; db_reaction = db_reaction->findNextBlock( "reaction" ) ){
      //get reaction rate params
      db_reaction->require("oxidizer_name",oxidizer_name);
      db_reaction->require("oxidizer_MW",oxidizer_MW);
      db_reaction->require("pre_exponential_factor",a);
      db_reaction->require("activation_energy",e);
      db_reaction->require("stoich_coeff_ratio",phi);
      db_reaction->require("heat_of_reaction_constant",hrxn);
      db_reaction->getWithDefault("use_co2co",use_co2co,false);
      _use_co2co_l.push_back(use_co2co);
      _MW_l.push_back(oxidizer_MW);
      _oxid_l.push_back(oxidizer_name);
      _a_l.push_back(a);
      _e_l.push_back(e);
      _phi_l.push_back(phi);
      _hrxn_l.push_back(hrxn);
      std::string rate_name = "char_reaction";
      std::stringstream str_l;
      str_l << _NUM_reactions;
      rate_name += str_l.str();
      std::stringstream str_qn;
      rate_name += "_qn";
      str_qn << d_quadNode;
      rate_name += str_qn.str();
      VarLabel* _reaction_rate_varlabel_temp = VarLabel::create( rate_name, CCVariable<double>::getTypeDescription() );
      _reaction_rate_varlabels.push_back(_reaction_rate_varlabel_temp);
      _NUM_reactions += 1;
    }
    if (_NUM_reactions == 0 ) {
      throw ProblemSetupException("Error: No SmithChar oxidation reactions specified in <ParticleProperties>.", __FILE__, __LINE__);
    }
  } else {
    throw ProblemSetupException("Error: SmithChar oxidation coefficients missing in <ParticleProperties>.", __FILE__, __LINE__);
  }

  diffusion_terms binary_diff_terms; // this allows access to the binary diff coefficients etc, in the header file.
  int table_size;
  table_size = binary_diff_terms.num_species;
  // find indices specified by user.
  std::vector<int> specified_indices;
  bool check_species;
  for (int j=0; j<_NUM_species; j++) {
    check_species = true;
    for (int i=0; i<table_size; i++) {
      if (_species_names[j]==binary_diff_terms.sp_name[i]){
        specified_indices.push_back(i);
        check_species = false;
      }
    }
    if (check_species ) {
      throw ProblemSetupException("Error: Species specified in SmithChar oxidation species, not found in SmithChar data-base (please add it).", __FILE__, __LINE__);
    }
  }
  std::vector<double> temp_v;
  for (int i=0; i<_NUM_species; i++) {
    temp_v.clear();
    _MW_species.push_back(binary_diff_terms.MW_sp[specified_indices[i]]);
    helper.add_lookup_species(_species_names[i]); // request all indicated species from table
    for (int j=0; j<_NUM_species; j++) {
      temp_v.push_back(binary_diff_terms.D_matrix[specified_indices[i]][specified_indices[j]]);
    }
    _D_mat.push_back(temp_v);
  }
  // find index of the oxidizers.
  for (int reac=0; reac<_NUM_reactions; reac++) {
    for (int spec=0; spec<_NUM_species; spec++) {
      if (_oxid_l[reac] == _species_names[spec]){
        _oxidizer_indices.push_back(spec);
      }
    }
  }
  // find index of the other species.
  for (int spec=0; spec<_NUM_species; spec++) {
    if(std::find(_oxidizer_indices.begin(), _oxidizer_indices.end(), spec) != _oxidizer_indices.end()){
        // spec is in _oxidizer_indices
    } else {
      _other_indices.push_back(spec);
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
CharOxidationSmithConstLv0::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CharOxidationSmithConstLv0::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidationSmithConstLv0::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_particletempLabel);
  tsk->computes(d_surfacerateLabel);
  tsk->computes(d_PO2surfLabel);
  for (std::vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  ApplicationInterface* m_application = sched->getApplication();
  
  if( m_application && m_application->getVisIt() && !initialized ) {
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name.
    ApplicationInterface::interactiveVar var;
    var.component  = "Arches";
    var.name       = "CharOx-PreExp-Factor-O2";
    var.type       = Uintah::TypeDescription::double_type;
    var.value      = (void *) &(_a_l[0]);
    var.range[0]   = -1.0e9;
    var.range[1]   = +1.0e9;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    // variable 2 - Must start with the component name and have NO
    // spaces in the var name.
    var.component  = "Arches";
    var.name       = "CharOx-Activation-Energy-O2";
    var.type       = Uintah::TypeDescription::double_type;
    var.value      = (void *) &(_e_l[0]);
    var.range[0]   = -1.0e9;
    var.range[1]   = +1.0e9;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    initialized = true;
  }
#endif
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidationSmithConstLv0::initVars( const ProcessorGroup * pc,
                              const PatchSubset    * patches,
                              const MaterialSubset * matls,
                              DataWarehouse        * old_dw,
                              DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> char_rate;
    CCVariable<double> gas_char_rate;
    CCVariable<double> particle_temp_rate;
    CCVariable<double> surface_rate;
    CCVariable<double> PO2surf_;
    std::vector< CCVariable<double> > reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.

    new_dw->allocateAndPut( char_rate, d_modelLabel, matlIndex, patch );
    char_rate.initialize(0.0);
    new_dw->allocateAndPut( gas_char_rate, d_gasLabel, matlIndex, patch );
    gas_char_rate.initialize(0.0);
    new_dw->allocateAndPut( particle_temp_rate, d_particletempLabel, matlIndex, patch );
    particle_temp_rate.initialize(0.0);
    new_dw->allocateAndPut(surface_rate, d_surfacerateLabel, matlIndex, patch );
    surface_rate.initialize(0.0);
    new_dw->allocateAndPut(PO2surf_, d_PO2surfLabel, matlIndex, patch );
    PO2surf_.initialize(0.0);
    for (int l=0; l<_NUM_reactions;l++ ){
      new_dw->allocateAndPut( reaction_rate_l[l], _reaction_rate_varlabels[l], matlIndex, patch );
      reaction_rate_l[l].initialize(0.0);
    }


  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
CharOxidationSmithConstLv0::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{


  // get gas phase temperature label
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmithConstLv0: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }
  // get species labels
  for (int l=0; l<_NUM_species; l++) {
    if (VarLabel::find(_species_names[l])) {
      VarLabel * _species_varlabel_temp = VarLabel::find(_species_names[l]);
      _species_varlabels.push_back(_species_varlabel_temp);
    } else {
      throw InvalidValue("ERROR: CharOxidationSmithConstLv0: problemSetup(): can't find gas phase oxidizer.",__FILE__,__LINE__);
    }
  }
  // get gas phase mixture_molecular_weight label
  if (VarLabel::find("mixture_molecular_weight")) {
    _MW_varlabel = VarLabel::find("mixture_molecular_weight");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmithConstLv0: problemSetup(): can't find gas phase mixture_molecular_weight.",__FILE__,__LINE__);
  }


  std::string taskname = "CharOxidationSmithConstLv0::computeModel";
  Task* tsk = scinew Task(taskname, this, &CharOxidationSmithConstLv0::computeModel, timeSubStep );

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_particletempLabel);
    tsk->computes(d_surfacerateLabel);
    tsk->computes(d_PO2surfLabel);
    for (std::vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
      tsk->computes(*iter);
    }
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_particletempLabel);
    tsk->modifies(d_surfacerateLabel);
    tsk->modifies(d_PO2surfLabel);
    for (std::vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
      tsk->modifies(*iter);
    }
    which_dw = Task::NewDW;
  }

  for (std::vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    tsk->requires( which_dw, *iter, gn, 0 );
  }
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 );
  tsk->requires( which_dw, _number_density_varlabel, gn, 0 );
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 );
  tsk->requires( which_dw, _char_varlabel, gn, 0 );

  for (int i=0; i<_nQn_part;i++ ){
  tsk->requires( which_dw, _length_varlabel[i], gn, 0 );
  tsk->requires( which_dw, _weight_varlabel[i], gn, 0 );
  }

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires( which_dw, _gas_temperature_varlabel, gn, 0);
  for (int l=0; l<_NUM_species; l++) {
    tsk->requires( which_dw, _species_varlabels[l], gn, 0 );
  }
  tsk->requires( which_dw, _MW_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, d_fieldLabels->d_delTLabel);
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, _RC_RHS_source_varlabel, gn, 0 );

  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0);
  tsk->requires( Task::NewDW, _devolRCLabel, gn, 0);
  if ( _char_birth_label != nullptr )
    tsk->requires( Task::NewDW, _char_birth_label, gn, 0 );
  if ( _rawcoal_birth_label != nullptr )
    tsk->requires( Task::NewDW, _rawcoal_birth_label, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CharOxidationSmithConstLv0::computeModel( const ProcessorGroup * pc,
                                  const PatchSubset    * patches,
                                  const MaterialSubset * matls,
                                  DataWarehouse        * old_dw,
                                  DataWarehouse        * new_dw,
                                  const int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    Vector Dx = patch->dCell();
    double vol = Dx.x()* Dx.y()* Dx.z();

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    double dt = DT;

    CCVariable<double> char_rate;
    CCVariable<double> gas_char_rate;
    CCVariable<double> particle_temp_rate;
    CCVariable<double> surface_rate;
    CCVariable<double> PO2surf_;
    std::vector< CCVariable<double> > reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.

    DenseMatrix* dfdrh = scinew DenseMatrix(_NUM_reactions,_NUM_reactions);

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( char_rate, d_modelLabel, matlIndex, patch );
      char_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_char_rate, d_gasLabel, matlIndex, patch );
      gas_char_rate.initialize(0.0);
      new_dw->allocateAndPut( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      particle_temp_rate.initialize(0.0);
      new_dw->allocateAndPut(surface_rate, d_surfacerateLabel, matlIndex, patch );
      surface_rate.initialize(0.0);
      new_dw->allocateAndPut(PO2surf_, d_PO2surfLabel, matlIndex, patch );
      PO2surf_.initialize(0.0);
      for (int l=0; l<_NUM_reactions;l++ ){
        new_dw->allocateAndPut( reaction_rate_l[l], _reaction_rate_varlabels[l], matlIndex, patch );
        reaction_rate_l[l].initialize(0.0);
      }
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( char_rate, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_char_rate, d_gasLabel, matlIndex, patch );
      new_dw->getModifiable( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      new_dw->getModifiable( surface_rate, d_surfacerateLabel, matlIndex, patch );
      new_dw->getModifiable( PO2surf_, d_PO2surfLabel, matlIndex, patch );
      for (int l=0; l<_NUM_reactions;l++ ){
        new_dw->getModifiable( reaction_rate_l[l], _reaction_rate_varlabels[l], matlIndex, patch );
      }
    }

    std::vector< constCCVariable<double> > old_reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.
    for (int l=0; l<_NUM_reactions;l++ ){
      which_dw->get( old_reaction_rate_l[l], _reaction_rate_varlabels[l], matlIndex, patch, gn, 0 );
    }

    constCCVariable<Vector> gasVel;
    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    constCCVariable<Vector> partVel;
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);
    constCCVariable<double> den;
    which_dw->get( den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> temperature;
    which_dw->get( temperature , _gas_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> particle_temperature;
    which_dw->get( particle_temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    std::vector< constCCVariable<double> > length(_nQn_part);
    std::vector< constCCVariable<double> > weight(_nQn_part);
    for (int i=0; i<_nQn_part;i++ ){
      which_dw->get( length[i], _length_varlabel[i], matlIndex, patch, gn, 0 );
      which_dw->get( weight[i], _weight_varlabel[i], matlIndex, patch, gn, 0 );
    }
    constCCVariable<double> rawcoal_mass;
    which_dw->get( rawcoal_mass, _rcmass_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> char_mass;
    which_dw->get( char_mass, _char_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> RHS_source;
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RC_RHS_source;
    new_dw->get( RC_RHS_source , _RC_RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> number_density;
    which_dw->get( number_density , _number_density_varlabel , matlIndex , patch , gn , 0 );
    std::vector< constCCVariable<double> > species(_NUM_species);
    for (int l=0; l<_NUM_species; l++) {
      which_dw->get( species[l], _species_varlabels[l], matlIndex, patch, gn, 0 );
    }
    constCCVariable<double> MWmix;
    which_dw->get( MWmix, _MW_varlabel, matlIndex, patch, gn, 0 );  // in kmol/kg_mix ?
    constCCVariable<double> devolRC;
    new_dw->get( devolRC, _devolRCLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> rawcoal_birth;
    constCCVariable<double> char_birth;
    bool add_rawcoal_birth = false;
    bool add_char_birth = false;
    if ( _rawcoal_birth_label != nullptr ){
      add_rawcoal_birth = true;
      new_dw->get( rawcoal_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 );
    }
    if ( _char_birth_label != nullptr ){
      add_char_birth = true;
      new_dw->get( char_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 );
    }

    // initialize all temporary variables which are use in the cell loop.
    int count;
    double relative_velocity;
    double CO_CO2_ratio;
    double CO2onCO;
    double gas_rho;
    double gas_T;
    double p_T;
    double p_diam;
    double p_area;
    double rc;
    double ch;
    double w;
    double MW;
    double d_mass;
    double h_rxn;
    double r_devol;
    double r_devol_ns;
    double RHS;
    double RHS_v;
    double Re_p;
    double cg;
    double residual;
    double char_mass_rate;
    double AreaSum;
    double surfaceAreaFraction;
    double sum_x_D;
    double sum_x;
    double delta;
    std::vector<double> phi_l(_NUM_reactions);
    std::vector<double> hrxn_l(_NUM_reactions);
    std::vector<double> oxid_mass_frac(_NUM_reactions);
    std::vector<double> oxid_mole_frac(_NUM_reactions);
    std::vector<double> co_r(_NUM_reactions);
    std::vector<double> k_r(_NUM_reactions);
    std::vector<double> species_mass_frac(_NUM_species);
    std::vector<double> rh_l(_NUM_reactions);
    std::vector<double> rh_l_new(_NUM_reactions);
    std::vector<double> _D_oxid_mix_l(_NUM_reactions);
    std::vector<double> F(_NUM_reactions);
    std::vector<double> F_delta(_NUM_reactions);
    std::vector<double> rh_l_delta(_NUM_reactions);
    std::vector<double> Sc(_NUM_reactions);
    std::vector<double> Sh(_NUM_reactions);

    InversionBase* invf;
    if (_NUM_reactions==2){
      invf = scinew invert_2_2;
    } else if (_NUM_reactions==3){
      invf = scinew invert_3_3;
    } else {
      throw InvalidValue("ERROR: CharOxidationSmithConstLv0: Matrix inversion not implemented for the number of reactions being used.",__FILE__,__LINE__);
    }

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for(range,  [&]( int i,  int j, int k){

      if (weight[d_quadNode](i,j,k)/_weight_scaling_constant < _weight_small) {
        char_rate(i,j,k) = 0.0;
        gas_char_rate(i,j,k) = 0.0;
        particle_temp_rate(i,j,k) = 0.0;
        surface_rate(i,j,k) = 0.0;
        for (int l=0; l<_NUM_reactions; l++) {
          reaction_rate_l[l](i,j,k)=0.0;
        }

      } else {
        // populate the temporary variables.
        Vector gas_vel = gasVel(i,j,k); // [m/s]
        Vector part_vel = partVel(i,j,k);// [m/s]
        gas_rho=den(i,j,k);// [kg/m^3]
        gas_T=temperature(i,j,k);// [K]
        p_T=particle_temperature(i,j,k);// [K]
        p_diam=length[d_quadNode](i,j,k);// [m]
        rc=rawcoal_mass(i,j,k);// [kg/#]
        ch=char_mass(i,j,k);// [kg/#]
        w=weight[d_quadNode](i,j,k);// [#/m^3]
        MW=1./MWmix(i,j,k); // [kg mix / kmol mix] (MW in table is 1/MW).
        RHS=RHS_source(i,j,k)*_char_scaling_constant*_weight_scaling_constant; // [kg/s]
        r_devol=devolRC(i,j,k)*_RC_scaling_constant*_weight_scaling_constant; // [kg/m^3/s]
        r_devol_ns=-r_devol; // [kg/m^3/s]
        RHS_v=RC_RHS_source(i,j,k)*_RC_scaling_constant*_weight_scaling_constant; // [kg/s]


        // populate temporary variable vectors
        delta = 1e-6;
        dfdrh->zero();// [-]
        for (int l=0; l<_NUM_reactions; l++) {
          rh_l[l]=old_reaction_rate_l[l](i,j,k);// [kg/m^3/s]
          rh_l_new[l]=old_reaction_rate_l[l](i,j,k);// [kg/m^3/s]
          // These variables are all set downstream.
          //F[l]=0.0;// [kg/m^3/s]
          //F_delta[l]=0.0;// [kg/m^3/s]
          //rh_l_delta[l]=0.0;// [kg/m^3/s]
          //Sc[l]=0.0;// [-]
          //Sh[l]=0.0;// [-]
          //_D_oxid_mix_l[l]=0.0;// [m^2/s]
          //oxid_mole_frac[l]=0.0;// [mole fraction]
          //co_r[l]=0.0;// [mole fraction]
          //k_r[l]=0.0;// [mole fraction]
        }
        for (int l=0; l<_NUM_reactions; l++) {
          oxid_mass_frac[l]=species[_oxidizer_indices[l]](i,j,k);// [mass fraction]
        }
        for (int l2=0; l2<_NUM_species; l2++) {
          species_mass_frac[l2]=species[l2](i,j,k);// [mass fraction]
        }

        // update the rate
        AreaSum = 0.0;
        for (int ix=0; ix<_nQn_part;ix++ ){
          AreaSum+=  weight[ix](i,j,k)*length[ix](i,j,k)*length[ix](i,j,k); // [#/m]
        }
        surfaceAreaFraction=w*p_diam*p_diam/AreaSum; // [-] this is the weighted area fraction for the current particle size.
        CO_CO2_ratio = 200.*exp(-9000./(_R_cal*p_T)); // [ kg CO / kg CO2]
        CO_CO2_ratio=CO_CO2_ratio*44.0/28.0; // [kmoles CO / kmoles CO2]
        CO2onCO=1./CO_CO2_ratio; // [kmoles CO2 / kmoles CO]
              for (int l=0; l<_NUM_reactions; l++) {
          phi_l[l] = (_use_co2co_l[l]) ? (CO2onCO + 1)/(CO2onCO + 0.5) : _phi_l[l]; 
          hrxn_l[l] = (_use_co2co_l[l]) ? (CO2onCO*_HF_CO2 + _HF_CO)/(1+CO2onCO) : _hrxn_l[l]; 
        }
        relative_velocity = sqrt( ( gas_vel.x() - part_vel.x() ) * ( gas_vel.x() - part_vel.x() ) +
                                         ( gas_vel.y() - part_vel.y() ) * ( gas_vel.y() - part_vel.y() ) +
                                         ( gas_vel.z() - part_vel.z() ) * ( gas_vel.z() - part_vel.z() )  );// [m/s]
        Re_p  = relative_velocity * p_diam / ( _dynamic_visc / gas_rho ); // Reynolds number [-]

        cg = _gasPressure / (_R * gas_T * 1000.); // [kmoles/m^3] - Gas concentration
        p_area = M_PI * p_diam*p_diam; // particle surface area [m^2]
        // Calculate oxidizer diffusion coefficient // effect diffusion through stagnant gas (see "Multicomponent Mass Transfer", Taylor and Krishna equation 6.1.14)
        for (int l=0; l<_NUM_reactions; l++) {
          sum_x_D=0;
          sum_x=0;
          for (int l2=0; l2<_NUM_species; l2++) {
            sum_x_D = (_oxid_l[l] != _species_names[l2]) ? sum_x_D + species_mass_frac[l2]/(_MW_species[l2]*_D_mat[_oxidizer_indices[l]][l2]) : sum_x_D;
            sum_x = (_oxid_l[l] != _species_names[l2]) ? sum_x + species_mass_frac[l2]/(_MW_species[l2]) : sum_x;
          }
          _D_oxid_mix_l[l] = sum_x/sum_x_D * std::sqrt(CUBE( gas_T/_T0));
        }
        for (int l=0; l<_NUM_reactions; l++) {
          Sc[l] = _dynamic_visc / (gas_rho * _D_oxid_mix_l[l]); // Schmidt number [-]
          Sh[l] = 2.0 + 0.6 * std::sqrt(Re_p) * std::cbrt(Sc[l]); // Sherwood number [-]
          oxid_mole_frac[l] = oxid_mass_frac[l] * MW / _MW_l[l]; // [mole fraction]
          co_r[l] = cg * oxid_mole_frac[l]; // oxidizer concentration, [kmoles/m^3]
          k_r[l] = ( 10.0 * _a_l[l] * exp( - _e_l[l] / ( _R_cal * p_T)) * _R * p_T * 1000.0) / ( _Mh * phi_l[l] * 101325. ); // [m / s]
        }
        // Newton-Raphson solve for rh_l.
        // rh_(n+1) = rh_(n) - (dF_(n)/drh_(n))^-1 * F_(n)
        count=0;
        for (int it=0; it<100; it++) {
          count=count+1;
          for (int l=0; l<_NUM_reactions; l++) {
            rh_l[l]=rh_l_new[l];
          }
          // get F and Jacobian -> dF/drh
          root_function( F, rh_l, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, w, p_area, _D_oxid_mix_l, phi_l );
          for (int j=0; j<_NUM_reactions; j++) {
            for (int k=0; k<_NUM_reactions; k++) {
              rh_l_delta[k] = rh_l[k];
            }
            rh_l_delta[j] = rh_l[j]+delta;
            root_function( F_delta, rh_l_delta, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, w, p_area, _D_oxid_mix_l, phi_l );
            for (int l=0; l<_NUM_reactions; l++) {
              (*dfdrh)[l][j] = (F_delta[l] - F[l]) / delta;
            }
          }
          // invert Jacobian -> (dF_(n)/drh_(n))^-1
          invf->invert_mat(dfdrh); // simple matrix inversion for a 2x2 matrix.
          // get rh_(n+1)
          double dominantRate=0.0;
          for (int l=0; l<_NUM_reactions; l++) {
            for (int var=0; var<_NUM_reactions; var++) {
              rh_l_new[l]-=(*dfdrh)[l][var]*F[var];
            }
              dominantRate=std::max(dominantRate,std::abs(rh_l_new[l]));
          }
          residual = 0.0;

          for (int l=0; l<_NUM_reactions; l++) {
            residual += std::abs(F[l])/dominantRate;
          }
          // make sure rh_(n+1) is inbounds
          for (int l=0; l<_NUM_reactions; l++) {
            rh_l_new[l]=std::min(0.01*_gasPressure, std::max(0.0, rh_l_new[l])); // max rate adjusted based on pressure (empirical limit)
          }
          if (residual < 1e-3) {
            break;
          }
        } // end newton solve
        if (count > 90){
            std::cout << "warning no solution found in char ox: [" << i << ", " << j << ", " << k << "] " << std::endl;
            std::cout << "gas_rho: " << gas_rho << std::endl;
            std::cout << "gas_T: " << gas_T << std::endl;
            std::cout << "p_T: " << p_T << std::endl;
            std::cout << "p_diam: " << p_diam << std::endl;
            std::cout << "relative_velocity: " << relative_velocity << std::endl;
            std::cout << "w: " << w << std::endl;
            std::cout << "MW: " << MW << std::endl;
            std::cout << "r_devol_ns: " << r_devol_ns << std::endl;
            std::cout << "oxid_mass_frac[0]: " << oxid_mass_frac[0] << std::endl;
            std::cout << "oxid_mass_frac[1]: " << oxid_mass_frac[1] << std::endl;
            std::cout << "_D_oxid_mix_l[0]: " << _D_oxid_mix_l[0] << std::endl;
            std::cout << "_D_oxid_mix_l[1]: " << _D_oxid_mix_l[1] << std::endl;
            std::cout << "rh_l_new[0]: " << rh_l_new[0] << std::endl;
            std::cout << "rh_l_new[1]: " << rh_l_new[1] << std::endl;
        }
        // convert rh units from kg/m^3/s to kg/s/#
        char_mass_rate  = 0.0;
              d_mass = 0.0;
              h_rxn = 0.0; // this is the reaction rate weighted heat of reaction. It is needed so we can used the clipped value when computed the heat of reaction rate.
                     // h_rxn = sum(hrxn_l * rxn_l)/sum(rxn_l)
        double oxi_lim = 0.0; // max rate due to reactions
        double rh_l_i = 0.0;
        for (int l=0; l<_NUM_reactions; l++) {
          reaction_rate_l[l](i,j,k)=rh_l_new[l];// [kg/m^3/s] this is for the intial guess during the next time-step (that is why it is before the initial clipping).
          // check to see if the reaction rate is oxidizer limited.
          oxi_lim = (oxid_mass_frac[l] * gas_rho * surfaceAreaFraction) / dt;// [kg/s/#] // here the surfaceAreaFraction parameter is allowing us to only consume the oxidizer multiplied by the weighted area fraction for the current particle.
          rh_l_i = std::min(rh_l_new[l], oxi_lim);
          char_mass_rate += -rh_l_i/w;// [kg/s/#]  negative sign because we are computing the destruction rate for the particles.
                d_mass += rh_l_i;
                h_rxn += hrxn_l[l] * rh_l_i;
        }
              h_rxn /= (d_mass + 1e-50); // [J/mole]
        
        // check to see if reaction rate is fuel limited.
        if ( add_rawcoal_birth && add_char_birth ){
          char_mass_rate = std::max( char_mass_rate, - ((rc+ch)/(dt) + (RHS + RHS_v)/(vol*w) + r_devol/w + char_birth(i,j,k)/w + rawcoal_birth(i,j,k)/w )); // [kg/s/#] 
        } else {
          char_mass_rate = std::max( char_mass_rate, - ((rc+ch)/(dt) + (RHS + RHS_v)/(vol*w) + r_devol/w )); // [kg/s/#] 
        }
        char_mass_rate = std::min( 0.0, char_mass_rate); // [kg/s/#] make sure we aren't creating char.
        char_rate(i,j,k) = (char_mass_rate*w)/(_char_scaling_constant*_weight_scaling_constant); // [kg/m^3/s - scaled]
        gas_char_rate(i,j,k) = -char_mass_rate*w;// [kg/m^3/s] (negative sign for exchange between solid and gas)
        particle_temp_rate(i,j,k) = h_rxn*char_mass_rate*w/_Mh*1000.; // [J/s/m^3] -- the *1000 is need to convert J/mole to J/kmole.
        surface_rate(i,j,k) = char_mass_rate/p_area;  // in [kg/(s # m^2)]
        PO2surf_(i,j,k) = 0.0; // multiple oxidizers, so we are leaving this empty.

      } // else statement
    }); //end cell loop
  delete invf;
  // delete scinew DenseMatrix
  delete dfdrh;
  } //end patch loop
}

inline void
CharOxidationSmithConstLv0::root_function( std::vector<double> &F, std::vector<double> &rh_l, std::vector<double> &co_r,
                                   double &gas_rho, double &cg, std::vector<double> &k_r, double &MW,
                                   double &r_devol, double &p_diam, std::vector<double> &Sh,
                                   double &w, double &p_area,
                                   std::vector<double> &_D_oxid_mix_l, std::vector<double> &phi_l ){

  double rh = 0.0;
  double Bjm = 0.0;
  double Fac = 0.0;
  double mtc_r = 0.0;
  double numerator = 0.0;
  double denominator = 0.0;
  for (int l=0; l<_NUM_reactions; l++) {
    rh = std::accumulate(rh_l.begin(), rh_l.end(), 0.0);
    Bjm = std::min( 80.0 , (rh + r_devol)/w/( 2. * M_PI * _D_oxid_mix_l[l] * p_diam * gas_rho ) ); // [-] // this is the derived for mass flux  BSL chapter 22
    Fac = ( Bjm >= 1e-7 ) ?  Bjm/(exp(Bjm)-1.) : 1.0; // also from BSL chapter 22 the mass transfer correction factor.
    mtc_r = (Sh[l] * _D_oxid_mix_l[l] * Fac) / p_diam; // [m/s]
    numerator =  SQUARE(p_area * w) * _Mh * MW * phi_l[l] * k_r[l] * mtc_r * _S * co_r[l] * cg; // [(#^2 kg-char kg-mix) / (s^2 m^6)]
    denominator = MW * p_area * w *cg * (k_r[l] * _S + mtc_r); // [(kg-mix #) / (m^3 s)]
    F[l] = rh_l[l] - numerator / ( denominator + rh  + r_devol); // [kg-char/m^3/s]
  }
}
