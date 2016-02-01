#include <CCA/Components/Arches/CoalModels/CharOxidationSmith.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <Core/Containers/StaticArray.h>
#include <spatialops/util/TimeLogger.h>
using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
CharOxidationSmithBuilder::CharOxidationSmithBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

CharOxidationSmithBuilder::~CharOxidationSmithBuilder(){}

ModelBase* CharOxidationSmithBuilder::build() {
  return scinew CharOxidationSmith( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationSmith::CharOxidationSmith( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: CharOxidation(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Set constants
  _pi = acos(-1.0);
  _R = 8.314; // J/K/mol
  _WO2 = 32.0; //g/mol
  _WCO2 = 44.0; //g/mol
  _WH2O = 18.0; //g/mol
  _WN2 = 28.0; //g/mol
  _small = 1e-30;
  // Enthalpy of formation (J/mol)
  _HF_CO2 = -393509.0;
  _HF_CO  = -110525.0;
  //binary diffsuion at 293 K
  _D1 = 0.153e-4; //O2-CO2 m^2/s
  _D2 = 0.240e-4; //O2-H2O
  _D3 = 0.219e-4; //O2-N2
  _T0 = 293.0;
  
  _char_birth_label = NULL;
  _rawcoal_birth_label = NULL; 

}

CharOxidationSmith::~CharOxidationSmith()
{
  for (vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    VarLabel::destroy( *iter );
  }
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CharOxidationSmith::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    std::string particleType;  
    db_coal_props->getAttribute("type",particleType);
    if (particleType != "coal"){
      throw InvalidValue("ERROR: CharOxidationSmith: Can't use particles of type: "+particleType,__FILE__,__LINE__);
    }
  
  // create raw coal mass var label 
  std::string rcmass_root = ParticleTools::parse_for_role_to_label(db, "raw_coal");
  std::string rcmass_name = ParticleTools::append_env( rcmass_root, d_quadNode );
  std::string rcmassqn_name = ParticleTools::append_qn_env(rcmass_root, d_quadNode );
  _rcmass_varlabel = VarLabel::find(rcmass_name);
  std::string rc_weighted_scaled_name = ParticleTools::append_qn_env( rcmass_root, d_quadNode );
  _rcmass_weighted_scaled_varlabel = VarLabel::find(rc_weighted_scaled_name); 
 
  // check for char mass and get scaling constant
  std::string char_root = ParticleTools::parse_for_role_to_label(db, "char");
  std::string char_name = ParticleTools::append_env( char_root, d_quadNode );
  std::string charqn_name = ParticleTools::append_qn_env( char_root, d_quadNode );
  _char_varlabel = VarLabel::find(char_name);
  std::string char_weighted_scaled_name = ParticleTools::append_qn_env( char_root, d_quadNode );
  _charmass_weighted_scaled_varlabel = VarLabel::find(char_weighted_scaled_name); 


  EqnBase& temp_char_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(charqn_name);
  DQMOMEqn& char_eqn = dynamic_cast<DQMOMEqn&>(temp_char_eqn);
  _char_scaling_constant = char_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = charqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  //CHAR get the birth term if any: 
  const std::string char_birth_name = char_eqn.get_model_by_type( "SimpleBirth" ); 
  std::string char_birth_qn_name = ParticleTools::append_qn_env(char_birth_name, d_quadNode); 
  if ( char_birth_name != "NULLSTRING" ){ 
    _char_birth_label = VarLabel::find( char_birth_qn_name ); 
  }

  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
  _RC_scaling_constant  = rcmass_eqn.getScalingConstant(d_quadNode)  ;   
  std::string RC_RHS = rcmassqn_name + "_RHS";  
  _RC_RHS_source_varlabel = VarLabel::find(RC_RHS);

  //RAW COAL get the birth term if any: 
  const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "SimpleBirth" ); 
  std::string rawcoal_birth_qn_name = ParticleTools::append_qn_env(rawcoal_birth_name, d_quadNode);
  if ( rawcoal_birth_name != "NULLSTRING" ){ 
    _rawcoal_birth_label = VarLabel::find( rawcoal_birth_qn_name ); 
  }

  // check for particle temperature 
  std::string temperature_root = ParticleTools::parse_for_role_to_label(db, "temperature"); 
  std::string temperature_name = ParticleTools::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
  if(_particle_temperature_varlabel == 0){
    throw ProblemSetupException("Error: Unable to find coal temperature label!!!! Looking for name: "+temperature_name, __FILE__, __LINE__); 
  }
 
  // check for length  
  _nQn_part = ParticleTools::get_num_env(db,ParticleTools::DQMOM); 
  std::string length_root = ParticleTools::parse_for_role_to_label(db, "size"); 
  for (int i=0; i<_nQn_part;i++ ){ 
    std::string length_name = ParticleTools::append_env( length_root, i ); 
    _length_varlabel.push_back(  VarLabel::find(length_name));
  }
  

  // get weight scaling constant
  std::string weightqn_name = ParticleTools::append_qn_env("w", d_quadNode); 
  for (int i=0; i<_nQn_part;i++ ){ 
  std::string weight_name = ParticleTools::append_env("w", i); 
    _weight_varlabel.push_back( VarLabel::find(weight_name) ); 
  }
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

  std::string number_density_name = ParticleTools::parse_for_role_to_label(db, "total_number_density");
  _number_density_varlabel = VarLabel::find(number_density_name); 

  // get Char source term label and devol label from the devolatilization model
  CoalModelFactory& modelFactory = CoalModelFactory::self(); 
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      _devolCharLabel = iModel->second->getCharSourceLabel();
      _devolRCLabel = iModel->second->getModelLabel() ;
    }
  }

  // Ensure the following species are populated from table
  // (this is expensive and should be avoided, if a species isn't needed)
  d_fieldLabels->add_species("temperature");
  //d_fieldLabels->add_species("O2");
  //d_fieldLabels->add_species("CO2");
  d_fieldLabels->add_species("H2O");
  d_fieldLabels->add_species("N2");
  d_fieldLabels->add_species("mixture_molecular_weight");
  
  
  // model global constants 
  _D_oxid_mix_l.push_back(0.00063);
  _D_oxid_mix_l.push_back(0.00049);
  _R_cal = 1.9872036; // [cal/ (K mol) ]
  _R = 8.314; // [J/ (K mol) ]
  // get model coefficients
  if (db_coal_props->findBlock("SmithChar")) {
    ProblemSpecP db_Smith = db_coal_props->findBlock("SmithChar");
    db_Smith->getWithDefault("use_simple_inversion",_use_simple_invert,false); 
    db_Smith->getWithDefault("char_MW",_Mh,12.0); // kg char / kmole char 
    db_Smith->getWithDefault("surface_area_mult_factor",_S,1.0);
    _NUM_reactions = 0;
    for ( ProblemSpecP db_reaction = db_Smith->findBlock( "reaction" ); db_reaction != 0; db_reaction = db_reaction->findNextBlock( "reaction" ) ){
      //get reaction rate params
      db_reaction->require("oxidizer_name",oxidizer_name);
      db_reaction->require("oxidizer_MW",oxidizer_MW);
      db_reaction->require("pre_exponential_factor",a);
      db_reaction->require("activation_energy",e);
      db_reaction->require("stoich_coeff_ratio",phi);
      _oxid_l.push_back(oxidizer_name);
      d_fieldLabels->add_species(oxidizer_name); // request oxidizer 
      _MW_l.push_back(oxidizer_MW);
      _oxid_l.push_back(oxidizer_name);
      _a_l.push_back(a);
      _e_l.push_back(e);
      _phi_l.push_back(phi);
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


}


//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
CharOxidationSmith::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CharOxidationSmith::initVars"; 
  Task* tsk = scinew Task(taskname, this, &CharOxidationSmith::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_particletempLabel);
  tsk->computes(d_surfacerateLabel);
  tsk->computes(d_PO2surfLabel);
  for (vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidationSmith::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> char_rate;
    CCVariable<double> gas_char_rate; 
    CCVariable<double> particle_temp_rate;
    CCVariable<double> surface_rate;
    CCVariable<double> PO2surf_;
    StaticArray< CCVariable<double> > reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.
    
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
CharOxidationSmith::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  // get gas phase temperature label 
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }
  // get oxidizer labels 
  for (int l=0; l<_NUM_reactions; l++) {
    if (VarLabel::find(_oxid_l[l])) {
      VarLabel * _oxidizer_varlabel_temp = VarLabel::find(_oxid_l[l]);
      _oxidizer_varlabels.push_back(_oxidizer_varlabel_temp);
    } else {
      throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase oxidizer.",__FILE__,__LINE__);
    }
  }
  // get gas phase CO2 label 
  //if (VarLabel::find("CO2")) {
  //  _CO2_varlabel = VarLabel::find("CO2");
  //} else {
  //  throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase CO2.",__FILE__,__LINE__);
  //}
  // get gas phase H2O label 
  if (VarLabel::find("H2O")) {
    _H2O_varlabel = VarLabel::find("H2O");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase H2O.",__FILE__,__LINE__);
  }
  // get gas phase N2 label 
  if (VarLabel::find("N2")) {
    _N2_varlabel = VarLabel::find("N2");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase N2.",__FILE__,__LINE__);
  }
  // get gas phase mixture_molecular_weight label 
  if (VarLabel::find("mixture_molecular_weight")) {
    _MW_varlabel = VarLabel::find("mixture_molecular_weight");
  } else {
    throw InvalidValue("ERROR: CharOxidationSmith: problemSetup(): can't find gas phase mixture_molecular_weight.",__FILE__,__LINE__);
  }

  std::string taskname = "CharOxidationSmith::sched_computeModel";
  Task* tsk = scinew Task(taskname, this, &CharOxidationSmith::computeModel, timeSubStep );

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_particletempLabel);
    tsk->computes(d_surfacerateLabel);
    tsk->computes(d_PO2surfLabel);
    for (vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
      tsk->computes(*iter);
    }
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_particletempLabel);
    tsk->modifies(d_surfacerateLabel);
    tsk->modifies(d_PO2surfLabel);
    for (vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
      tsk->modifies(*iter);
    }
    which_dw = Task::NewDW; 
  }

  for (vector<const VarLabel*>::iterator iter = _reaction_rate_varlabels.begin(); iter != _reaction_rate_varlabels.end(); iter++) {
    tsk->requires( which_dw, *iter, gn, 0 );
  }
  tsk->requires( Task::NewDW, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( Task::NewDW, _number_density_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _charmass_weighted_scaled_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _rcmass_weighted_scaled_varlabel, gn, 0 ); 

  for (int i=0; i<_nQn_part;i++ ){ 
  tsk->requires( which_dw, _length_varlabel[i], gn, 0 ); 
  tsk->requires( which_dw, _weight_varlabel[i], gn, 0 ); 
  }

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires( which_dw, _gas_temperature_varlabel, gn, 0);
  for (int l=0; l<_NUM_reactions; l++) {
    tsk->requires( which_dw, _oxidizer_varlabels[l], gn, 0 );
  }
  tsk->requires( which_dw, _H2O_varlabel, gn, 0 );
  tsk->requires( which_dw, _N2_varlabel, gn, 0 );
  tsk->requires( which_dw, _MW_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label()); 
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 ); 
  tsk->requires( Task::NewDW, _RC_RHS_source_varlabel, gn, 0 ); 
  
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0);
  tsk->requires( Task::NewDW, _devolCharLabel, gn, 0);
  tsk->requires( Task::NewDW, _devolRCLabel, gn, 0);
  if ( _char_birth_label != NULL )
    tsk->requires( Task::NewDW, _char_birth_label, gn, 0 ); 
  if ( _rawcoal_birth_label != NULL ) 
    tsk->requires( Task::NewDW, _rawcoal_birth_label, gn, 0 ); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CharOxidationSmith::computeModel( const ProcessorGroup * pc, 
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
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    Vector Dx = patch->dCell(); 
    double vol = Dx.x()* Dx.y()* Dx.z(); 
    
    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;

    CCVariable<double> char_rate;
    CCVariable<double> gas_char_rate; 
    CCVariable<double> particle_temp_rate;
    CCVariable<double> surface_rate;
    CCVariable<double> PO2surf_;
    StaticArray< CCVariable<double> > reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.
    dfdrh = scinew DenseMatrix(_NUM_reactions,_NUM_reactions);
     
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

    StaticArray< constCCVariable<double> > old_reaction_rate_l(_NUM_reactions); // char reaction rate for lth reaction.
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
    new_dw->get( particle_temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    StaticArray< constCCVariable<double> > length(_nQn_part); 
    StaticArray< constCCVariable<double> > weight(_nQn_part);
    for (int i=0; i<_nQn_part;i++ ){ 
      which_dw->get( length[i], _length_varlabel[i], matlIndex, patch, gn, 0 );
      which_dw->get( weight[i], _weight_varlabel[i], matlIndex, patch, gn, 0 );
    }
    constCCVariable<double> rawcoal_mass;
    which_dw->get( rawcoal_mass, _rcmass_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> char_mass;
    which_dw->get( char_mass, _char_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> rawcoal_weighted_scaled; 
    which_dw->get( rawcoal_weighted_scaled, _rcmass_weighted_scaled_varlabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> char_weighted_scaled; 
    which_dw->get( char_weighted_scaled, _charmass_weighted_scaled_varlabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> RHS_source; 
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RC_RHS_source; 
    new_dw->get( RC_RHS_source , _RC_RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> number_density; 
    new_dw->get( number_density , _number_density_varlabel , matlIndex , patch , gn , 0 );
    StaticArray< constCCVariable<double> > oxidizers(_NUM_reactions); 
    for (int l=0; l<_NUM_reactions; l++) {
      which_dw->get( oxidizers[l], _oxidizer_varlabels[l], matlIndex, patch, gn, 0 );
    }
    constCCVariable<double> H2O;
    which_dw->get( H2O, _H2O_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> N2;
    which_dw->get( N2, _N2_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> MWmix;
    which_dw->get( MWmix, _MW_varlabel, matlIndex, patch, gn, 0 );  // in kmol/kg_mix ?
    constCCVariable<double> devolChar;
    new_dw->get( devolChar, _devolCharLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> devolRC;
    new_dw->get( devolRC, _devolRCLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> rawcoal_birth; 
    constCCVariable<double> char_birth; 
    bool add_rawcoal_birth = false; 
    bool add_char_birth = false; 
    if ( _rawcoal_birth_label != NULL ){ 
      add_rawcoal_birth = true; 
      new_dw->get( rawcoal_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 ); 
    }
    if ( _char_birth_label != NULL ){ 
      add_char_birth = true; 
      new_dw->get( char_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 ); 
    }

    // initialize all temporary variables which are use in the cell loop.  
    int count; 
    double relative_velocity;
    double CO_CO2_ratio;
    double gas_rho;
    double gas_T;
    double p_T;
    double p_diam;
    double p_area;
    double rc;
    double ch;
    double w;
    double H2O_mass_frac;
    double N2_mass_frac;
    double MW;
    double r_devol;
    double RHS;
    double RHS_v;
    double dynamic_visc; 
    double Re_p; 
    double cg; 
    double residual; 
    double char_mass_rate; 
    std::vector<double> oxid_mass_frac;
    std::vector<double> rh_l;
    std::vector<double> rh_l_new;
    std::vector<double> _D_oxid_mix_l;
    std::vector<double> F;
    std::vector<double> Sc;
    std::vector<double> Sh;
    //SpatialOps::TimeLogger my_timer;
    //my_timer.start("my_total");

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
 
      if (weight[d_quadNode][c]/_weight_scaling_constant < _weight_small) {
        char_rate[c] = 0.0;
        gas_char_rate[c] = 0.0;
        particle_temp_rate[c] = 0.0;
        surface_rate[c] = 0.0;
        for (int l=0; l<_NUM_reactions; l++) {
          reaction_rate_l[l][c]=0.0;
        }

      } else {
        // populate the temporary variables.
        Vector gas_vel = gasVel[c]; // [m/s]
        Vector part_vel = partVel[c];// [m/s]
        gas_rho=den[c];// [kg/m^3]
        gas_T=temperature[c];// [K]
        p_T=particle_temperature[c];// [K]
        p_diam=length[d_quadNode][c];// [m]
        rc=rawcoal_mass[c];// [kg/#]
        ch=char_mass[c];// [kg/#]
        w=weight[d_quadNode][c];// [#/m^3]
        H2O_mass_frac=H2O[c];// [mass fraction]
        N2_mass_frac=N2[c];// [mass fraction]
        MW=1/MWmix[c]; // [kg mix / kmol mix] (MW in table is 1/MW).
        RHS=RHS_source[c]*_char_scaling_constant*_weight_scaling_constant; // [kg/s]
        r_devol=devolRC[c]*_char_scaling_constant*_weight_scaling_constant; // [kg/m^3/s]
        RHS_v=RC_RHS_source[c]*_char_scaling_constant*_weight_scaling_constant; // [kg/s]
        dynamic_visc = 0.00005; // [kg/(m s)] 

        
        // TODO list ....
        // get dynamice viscosity from ups file
        // incorporate some sort of diffusion model to get oxidizer diffusion coefficients. 
        // clean up all unused variables in .cc file
        // clean up all unused variables in .h file
        // do one last verification against matlab code to make sure things are working properly.

        // clear temporary variable vectors.
        _D_oxid_mix_l.clear();
        oxid_mass_frac.clear();
        rh_l.clear();
        rh_l_new.clear();
        F.clear();
        Sc.clear();
        Sh.clear();

        // populate temporary variable vectors
        dfdrh->zero();// [-]
        _D_oxid_mix_l.push_back(0.00063);// [m^2/s]
        _D_oxid_mix_l.push_back(0.00049);// [m^2/s]
        for (int l=0; l<_NUM_reactions; l++) {
          oxid_mass_frac.push_back(oxidizers[l][c]);// [mass fraction]
          rh_l.push_back(old_reaction_rate_l[l][c]);// [kg/m^3/s]
          rh_l_new.push_back(old_reaction_rate_l[l][c]);// [kg/m^3/s]
          F.push_back(0.0);// [kg/m^3/s] 
          Sc.push_back(0.0);// [-]
          Sh.push_back(0.0);// [-]
        }
        
        // update the rate 
        CO_CO2_ratio = 200*exp(-9000/(_R_cal*p_T)); // [ kg CO / kg CO2]
        CO_CO2_ratio=CO_CO2_ratio*44.0/28.0; // [kmoles CO / kmoles CO2]
        relative_velocity = sqrt( ( gas_vel.x() - part_vel.x() ) * ( gas_vel.x() - part_vel.x() ) + 
                                         ( gas_vel.y() - part_vel.y() ) * ( gas_vel.y() - part_vel.y() ) +
                                         ( gas_vel.z() - part_vel.z() ) * ( gas_vel.z() - part_vel.z() )  );// [m/s]  
        Re_p  = relative_velocity * p_diam / ( dynamic_visc / gas_rho ); // Reynolds number [-]
        
        cg = 101325 / (_R * p_T * 1000); // [kmoles/m^3] - Gas concentration 
        p_area; // particle surface area [m^2]
        p_area = _pi * pow(p_diam,2); // particle surface area [m^2]
        for (int l=0; l<_NUM_reactions; l++) {
          Sc[l] = dynamic_visc / (gas_rho * _D_oxid_mix_l[l]); // Schmidt number [-]
          Sh[l] = 2.0 + 0.6 * pow(Re_p,0.5) * pow(Sc[l],0.33333); // Sherwood number [-]
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
          root_function( F, dfdrh, rh_l, p_T, cg, oxid_mass_frac, MW, r_devol, gas_rho, p_diam, Sh, w, p_area );  
          // invert Jacobian -> (dF_(n)/drh_(n))^-1 
          if (_use_simple_invert){
            invert_2_2(dfdrh); // simple matrix inversion for a 2x2 matrix. 
          } else {
            dfdrh->invert(); // Lapack matrix inversion. 
          }
          // get rh_(n+1)
          for (int l=0; l<_NUM_reactions; l++) {
            rh_l_new[l]=rh_l[l];
            for (int var=0; var<_NUM_reactions; var++) {
              rh_l_new[l]-=(*dfdrh)[l][var]*F[var];
            }
          }
          // make sure rh_(n+1) is inbounds
          for (int l=0; l<_NUM_reactions; l++) {
            rh_l_new[l]=std::min(0.0, max(-1000.0, rh_l_new[l])); // rh can't be larger than zero or smaller than -1000
          }
          // see if solution is converged (F = 0.0)
          residual = 0.0; 
          for (int l=0; l<_NUM_reactions; l++) {
            residual += std::abs(F[l]);
          }
          if (residual < 1e-15) {
            break;
          }
        }
        // convert rh units from kg/m^3/s to kg/s/#
        char_mass_rate  = 0.0;
        for (int l=0; l<_NUM_reactions; l++) {
          reaction_rate_l[l][c]=rh_l_new[l];// [kg/m^3/s]
          char_mass_rate+=rh_l_new[l]/w;// [kg/s/#]
        }
        // check to see if reaction rate is oxidizer limited.
        for (int l=0; l<_NUM_reactions; l++) {
          char_mass_rate = max( char_mass_rate, - (oxid_mass_frac[l] * gas_rho) / (dt * w) );// [kg/s/#]  
        }
        // check to see if reaction rate is fuel limited.
        if ( add_rawcoal_birth && add_char_birth ){ 
          char_mass_rate = max( char_mass_rate, - ((rc+ch)/(dt) + (RHS + RHS_v)/(vol*w) + r_devol/w + char_birth[c]/w + rawcoal_birth[c]/w )); // [kg/s/#] equation assumes RC_scaling=Char_scaling
        } else { 
          char_mass_rate = max( char_mass_rate, - ((rc+ch)/(dt) + (RHS + RHS_v)/(vol*w) + r_devol/w )); // [kg/s/#] equation assumes RC_scaling=Char_scaling
        }
        char_mass_rate = min( 0.0, char_mass_rate); // [kg/s/#] make sure we aren't creating char.
        char_rate[c] = (char_mass_rate*w)/(_char_scaling_constant*_weight_scaling_constant); // [kg/m^3/s - scaled]
        gas_char_rate[c] = -char_mass_rate*w;// [kg/m^3/s] (negative sign for exchange between solid and gas)
        particle_temp_rate[c] = char_mass_rate*w/_Mh/(1.0+CO_CO2_ratio)*(CO_CO2_ratio*_HF_CO + _HF_CO2)*1000; // [J/s/m^3] -- the *1000 is need to convert J/mole to J/kmole.
        surface_rate[c] = char_mass_rate/p_area;  // in [kg/(s # m^2)]
        PO2surf_[c] = 0.0; // multiple oxidizers, so we are leaving this empty.

      }// else statement 
    }//end cell loop
  //my_timer.stop("my_total");
  // delete scinew DenseMatrix 
  delete dfdrh; 
  }//end patch loop
}

inline void 
CharOxidationSmith::root_function( std::vector<double> &F, DenseMatrix* &dfdrh, std::vector<double> &rh_l, double &p_T, double &cg, std::vector<double> &oxid_mass_frac, double &MW, double &r_devol, double &gas_rho, double &p_diam, std::vector<double> &Sh, double &w, double &p_area ){

  double oxid_mole_frac = 0.0;
  double co_r = 0.0;
  double k_r = 0.0;
  double rh = 0.0;
  double rtotal = 0.0;
  double Bjm = 0.0;
  double Fac = 0.0;
  double Bjmj = 0.0;
  double Denom = 0.0;
  double mtc_r = 0.0;
  double numerator = 0.0;
  double denominator = 0.0;
  for (int l=0; l<_NUM_reactions; l++) {
    oxid_mole_frac = oxid_mass_frac[l] * MW / _MW_l[l]; // [mole fraction]
    co_r = cg * oxid_mole_frac; // oxidizer concentration, [kmoles/m^3]
    k_r = ( 10 * _a_l[l] * exp( - _e_l[l] / ( _R_cal * p_T)) * _R * p_T * 1000) / ( _Mh * _phi_l[l] * 101325 ); // [m / s]
    rh = std::accumulate(rh_l.begin(), rh_l.end(), 0.0);
    rtotal = rh + r_devol; // [kg/m^3/s]
    Bjm = rtotal/( 2 * _pi * _D_oxid_mix_l[l] * gas_rho * p_diam ); // [m^3]
    Fac = 1.0; // [-]
    Bjmj = 0.0;
    Denom = 1.0; // [m^3] ??
    if (Bjm >= 1e-7) {
      Bjmj = Bjm;
      Denom = (Bjm >= 80) ? 1e25 : exp(Bjmj)-1;
      Fac = Bjm/Denom;
    } 
    mtc_r = (Sh[l] * _D_oxid_mix_l[l] * Fac) / p_diam; // [m/s]
    numerator = pow( p_area * w, 2.0) * _Mh * MW * _phi_l[l] * k_r * mtc_r * _S * co_r * cg; // [(#^2 kg-char kg-mix) / (s^2 m^6)] 
    denominator = MW * p_area * w *cg * (k_r * _S + mtc_r); // [(kg-mix #) / (m^3 s)]
    F[l] = rh_l[l] + numerator / ( denominator + rh  + r_devol); // [kg-char/m^3/s]
    for (int var=0; var<_NUM_reactions; var++) {
      (*dfdrh)[l][var] = (var==l) ? 1 - numerator/pow(denominator + rh + r_devol,2.0) : - numerator/pow(denominator + rh + r_devol,2.0); // [-]
    }
  } 
}

inline void 
CharOxidationSmith::invert_2_2( DenseMatrix* &dfdrh ){
  double a11=(*dfdrh)[0][0];
  double a12=(*dfdrh)[0][1];
  double a21=(*dfdrh)[1][0];
  double a22=(*dfdrh)[1][1];
  double det = a11*a22-a12*a21;
  double det_inv = 1/det;
  (*dfdrh)[0][0]=a22*det_inv;
  (*dfdrh)[0][1]=-a12*det_inv;
  (*dfdrh)[1][0]=-a21*det_inv;
  (*dfdrh)[1][1]=a11*det_inv;
}
