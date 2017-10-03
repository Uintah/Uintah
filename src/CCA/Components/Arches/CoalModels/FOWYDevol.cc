#include <CCA/Components/Arches/CoalModels/FOWYDevol.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/Utility/InverseErrorFunction.h>

#include <sci_defs/visit_defs.h>

//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
FOWYDevolBuilder::FOWYDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

FOWYDevolBuilder::~FOWYDevolBuilder(){}

ModelBase* FOWYDevolBuilder::build() {
  return scinew FOWYDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

FOWYDevol::FOWYDevol( std::string modelName,
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames,
                                              vector<std::string> scalarLabelNames,
                                              int qn )
: Devolatilization(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = acos(-1.0);

  std::string v_inf_name = ParticleTools::append_env( "v_inf", qn );
  _v_inf_label = VarLabel::create( v_inf_name, CCVariable<double>::getTypeDescription() );
  _rawcoal_birth_label = nullptr;

}

FOWYDevol::~FOWYDevol()
{

  VarLabel::destroy(_v_inf_label);

}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
FOWYDevol::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

    std::string particleType;
    db_coal_props->getAttribute("type",particleType);
    if (particleType != "coal"){
      throw InvalidValue("ERROR: FOWYDevol: Can't use particles of type: "+particleType,__FILE__,__LINE__);
    }

  // create raw coal mass var label and get scaling constant
  std::string rcmass_root = ParticleTools::parse_for_role_to_label(db, "raw_coal");
  std::string rcmass_name = ParticleTools::append_env( rcmass_root, d_quadNode );
  std::string rcmassqn_name = ParticleTools::append_qn_env( rcmass_root, d_quadNode );
  _rcmass_varlabel = VarLabel::find(rcmass_name);
  _rcmass_weighted_scaled_varlabel = VarLabel::find(rcmassqn_name);


  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
   _rc_scaling_constant = rcmass_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = rcmassqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);



  //RAW COAL get the birth term if any:
  const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
  std::string rawcoal_birth_qn_name = ParticleTools::append_qn_env(rawcoal_birth_name, d_quadNode);
  if ( rawcoal_birth_name != "NULLSTRING" ){
    _rawcoal_birth_label = VarLabel::find( rawcoal_birth_qn_name );
  }

  // create char mass var label
  std::string char_root = ParticleTools::parse_for_role_to_label(db, "char");
  std::string char_name = ParticleTools::append_env( char_root, d_quadNode );
  _char_varlabel = VarLabel::find(char_name);
  std::string char_weighted_scaled_name = ParticleTools::append_qn_env( char_root, d_quadNode );
  _charmass_weighted_scaled_varlabel = VarLabel::find(char_weighted_scaled_name);

  // check for char mass and get scaling constant
  std::string charqn_name = ParticleTools::append_qn_env( char_root, d_quadNode );

  std::string char_ic_RHS = charqn_name+"_RHS";
  _char_RHS_source_varlabel = VarLabel::find(char_ic_RHS);

  // create particle temperature label
  std::string temperature_root = ParticleTools::parse_for_role_to_label(db, "temperature");
  std::string temperature_name = ParticleTools::append_env( temperature_root, d_quadNode );
  _particle_temperature_varlabel = VarLabel::find(temperature_name);

  // Look for required scalars
  if (db_coal_props->findBlock("FOWYDevol")) {
    ProblemSpecP db_BT = db_coal_props->findBlock("FOWYDevol");
    db_BT->require("Tig", _Tig);
    db_BT->require("Ta", _Ta);
    db_BT->require("A", _A);
    db_BT->require("v_hiT", _v_hiT); // this is a
    db_BT->require("Tbp_graphite", _Tbp_graphite); // 
    db_BT->require("T_mu", _T_mu); // 
    db_BT->require("T_sigma", _T_sigma); // 
    db_BT->require("T_hardened_bond", _T_hardened_bond); // 
    db_BT->require("sigma", _sigma)  ;

  } else {
    throw ProblemSetupException("Error: FOWY coefficients missing in <CoalProperties>.", __FILE__, __LINE__);
  }
  if ( db_coal_props->findBlock("density")){
    db_coal_props->require("density", rhop);
  } else {
    throw ProblemSetupException("Error: You must specify density in <CoalProperties>.", __FILE__, __LINE__);
  }
  if ( db_coal_props->findBlock("diameter_distribution")){
    db_coal_props->require("diameter_distribution", particle_sizes);
  } else {
    throw ProblemSetupException("Error: You must specify diameter_distribution in <CoalProperties>.", __FILE__, __LINE__);
  }
  if ( db_coal_props->findBlock("ultimate_analysis")){
    ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis");
    CoalAnalysis coal;
    db_ua->require("C",coal.C);
    db_ua->require("H",coal.H);
    db_ua->require("O",coal.O);
    db_ua->require("N",coal.N);
    db_ua->require("S",coal.S);
    db_ua->require("H2O",coal.H2O);
    db_ua->require("ASH",coal.ASH);
    db_ua->require("CHAR",coal.CHAR);
    total_rc=coal.C+coal.H+coal.O+coal.N+coal.S; // (C+H+O+N+S) dry ash free total
    total_dry=coal.C+coal.H+coal.O+coal.N+coal.S+coal.ASH+coal.CHAR; // (C+H+O+N+S+char+ash)  moisture free total
    rc_mass_frac=total_rc/total_dry; // mass frac of rc (dry)
    char_mass_frac=coal.CHAR/total_dry; // mass frac of char (dry)
    ash_mass_frac=coal.ASH/total_dry; // mass frac of ash (dry)
    int p_size=particle_sizes.size();
    for (int n=0; n<p_size; n=n+1)
      {
        vol_dry.push_back((pi/6)*std::pow(particle_sizes[n],3.0)); // m^3/particle
        mass_dry.push_back(vol_dry[n]*rhop); // kg/particle
        ash_mass_init.push_back(mass_dry[n]*ash_mass_frac); // kg_ash/particle (initial)
        char_mass_init.push_back(mass_dry[n]*char_mass_frac); // kg_char/particle (initial)
        rc_mass_init.push_back(mass_dry[n]*rc_mass_frac); // kg_ash/particle (initial)
      }
  } else {
    throw ProblemSetupException("Error: You must specify ultimate_analysis in <CoalProperties>.", __FILE__, __LINE__);
  }

  // get weight scaling constant
  std::string weightqn_name = ParticleTools::append_qn_env("w", d_quadNode);
  std::string weight_name = ParticleTools::append_env("w", d_quadNode);
  _weight_varlabel = VarLabel::find(weight_name);
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);


#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( d_sharedState->getVisIt() && !initialized ) {
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name.
    SimulationState::interactiveVar var;
    var.name     = "Arches-Devol-Ultimate-Yield";
    var.type     = Uintah::TypeDescription::double_type;
    var.value    = (void *) &( _v_hiT);
    var.range[0]   = -1.0e9;
    var.range[1]   = +1.0e9;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    d_sharedState->d_UPSVars.push_back( var );

    initialized = true;
  }
#endif
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
FOWYDevol::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "FOWYDevol::initVars";
  Task* tsk = scinew Task(taskname, this, &FOWYDevol::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_charLabel);
  tsk->computes(_v_inf_label);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
FOWYDevol::initVars( const ProcessorGroup * pc,
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

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate;
    CCVariable<double> char_rate;
    CCVariable<double> v_inf;

    new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
    devol_rate.initialize(0.0);
    new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch );
    gas_devol_rate.initialize(0.0);
    new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
    char_rate.initialize(0.0);
    new_dw->allocateAndPut( v_inf, _v_inf_label, matlIndex, patch );
    v_inf.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
FOWYDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "FOWYDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &FOWYDevol::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;

  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_charLabel);
    tsk->computes(_v_inf_label);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_charLabel);
    tsk->modifies(_v_inf_label);
    which_dw = Task::NewDW;
  }
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 );
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 );
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 );
  tsk->requires( which_dw, _rcmass_weighted_scaled_varlabel, gn, 0 );
  tsk->requires( which_dw, _charmass_weighted_scaled_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label());
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, _char_RHS_source_varlabel, gn, 0 );
  if ( _rawcoal_birth_label != nullptr )
    tsk->requires( Task::NewDW, _rawcoal_birth_label, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
FOWYDevol::computeModel( const ProcessorGroup * pc,
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

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate;
    CCVariable<double> char_rate;
    CCVariable<double> v_inf;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch );
      gas_devol_rate.initialize(0.0);
      new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
      char_rate.initialize(0.0);
      new_dw->allocateAndPut( v_inf, _v_inf_label, matlIndex, patch );
      v_inf.initialize(0.0);
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch );
      new_dw->getModifiable( char_rate, d_charLabel, matlIndex, patch );
      new_dw->getModifiable( v_inf, _v_inf_label, matlIndex, patch );
    }

    constCCVariable<double> temperature;
    which_dw->get( temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> rcmass;
    which_dw->get( rcmass    , _rcmass_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> charmass;
    which_dw->get( charmass , _char_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> weight;
    which_dw->get( weight , _weight_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RHS_source;
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> char_RHS_source;
    new_dw->get( char_RHS_source , _char_RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> rc_weighted_scaled;
    which_dw->get( rc_weighted_scaled, _rcmass_weighted_scaled_varlabel, matlIndex , patch , gn , 0 );
    constCCVariable<double> char_weighted_scaled;
    which_dw->get( char_weighted_scaled, _charmass_weighted_scaled_varlabel, matlIndex , patch , gn , 0 );

    constCCVariable<double> rawcoal_birth;
    bool add_birth = false;
    if ( _rawcoal_birth_label != nullptr ){
      add_birth = true;
      new_dw->get( rawcoal_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 );
    }


    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    Uintah::parallel_for( range, [&] (int i, int j, int k) {
       double rcmass_init = rc_mass_init[d_quadNode];
       double Z=0;
   
       if (weight(i,j,k)/_weight_scaling_constant > _weight_small) {
   
         double rcmassph=rcmass(i,j,k);
         double RHS_sourceph=RHS_source(i,j,k);
         double temperatureph=temperature(i,j,k);
         double charmassph=charmass(i,j,k);
         double weightph=weight(i,j,k);
   
         //VERIFICATION
         //rcmassph=1;
         //temperatureph=300;
         //charmassph=0.0;
         //weightph=_rc_scaling_constant*_weight_scaling_constant;
         //rcmass_init = 1;
   
   
         // m_init = m_residual_solid + m_h_off_gas + m_vol
         // m_vol = m_init - m_residual_solid - m_h_off_gas
         // but m_h_off_gas = - m_char
         // m_vol = m_init - m_residual_solid + m_char
   
         double m_vol = rcmass_init - (rcmassph+charmassph);
         double v_inf_local = 0.5*_v_hiT*(1.0 + std::erf( (temperatureph - _T_mu) / (std::sqrt(2.0) * _T_sigma)));
         v_inf_local += (temperatureph > _T_hardened_bond) ? (temperatureph - _T_hardened_bond)/_Tbp_graphite : 0.0; // linear contribution
         v_inf_local = std::min(1.0,v_inf_local);
         // above hardened bond temperature.
         v_inf(i,j,k) = v_inf_local; 
         double f_drive = std::max((rcmass_init*v_inf_local - m_vol) , 0.0);
         double zFact =std::min(std::max(f_drive/rcmass_init/_v_hiT,2.5e-5 ),1.0-2.5e-5  );
   
         double rateMax = 0.0; 
         if ( add_birth ){ 
           rateMax = std::max(f_drive/dt 
               + (  (RHS_sourceph+char_RHS_source(i,j,k)) /vol + rawcoal_birth(i,j,k) ) / weightph
               * _rc_scaling_constant*_weight_scaling_constant , 0.0 );
         } else { 
           rateMax = std::max(f_drive/dt 
               + (  (RHS_sourceph+char_RHS_source(i,j,k)) /vol ) / weightph
               * _rc_scaling_constant*_weight_scaling_constant , 0.0 );
         }
         Z = std::sqrt(2.0) * erfinv(1.0-2.0*zFact );
         
         double rate = std::min(_A*exp(-(_Ta + Z *_sigma)/temperatureph)*f_drive , rateMax);
         devol_rate(i,j,k) = -rate*weightph/(_rc_scaling_constant*_weight_scaling_constant); //rate of consumption of raw coal mass
         gas_devol_rate(i,j,k) = rate*weightph; // rate of creation of coal off gas
         char_rate(i,j,k) = 0; // rate of creation of char
       } else {
         devol_rate(i,j,k) = 0;
         gas_devol_rate(i,j,k) = 0;
         char_rate(i,j,k) = 0;
       }
     } );
//end cell loop

  }//end patch loop
}
