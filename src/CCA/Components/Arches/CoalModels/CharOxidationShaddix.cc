#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
CharOxidationShaddixBuilder::CharOxidationShaddixBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

CharOxidationShaddixBuilder::~CharOxidationShaddixBuilder(){}

ModelBase* CharOxidationShaddixBuilder::build() {
  return scinew CharOxidationShaddix( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationShaddix::CharOxidationShaddix( std::string modelName,
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
  _WC = 12.0e-3;  //kg/mol
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

  _char_birth_label = nullptr;
  _rawcoal_birth_label = nullptr;

}

CharOxidationShaddix::~CharOxidationShaddix()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CharOxidationShaddix::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    std::string particleType;
    db_coal_props->getAttribute("type",particleType);
    if (particleType != "coal"){
      throw InvalidValue("ERROR: CharOxidationShaddix: Can't use particles of type: "+particleType,__FILE__,__LINE__);
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
  const std::string char_birth_name = char_eqn.get_model_by_type( "BirthDeath" );
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
  const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
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

  // get model coefficients
  if (db_coal_props->findBlock("ShaddixChar")) {
    ProblemSpecP db_Shad = db_coal_props->findBlock("ShaddixChar");
    //get reaction rate params
    db_Shad->require("As",_As);
    db_Shad->require("Es",_Es);
    db_Shad->require("n",_n);
  } else {
    throw ProblemSetupException("Error: ShaddixChar oxidation coefficients missing in <ParticleProperties>.", __FILE__, __LINE__);
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

  // get Char source term label and devol lable from the devolatilization model
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
  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species("temperature");
  helper.add_lookup_species("O2");
  helper.add_lookup_species("CO2");
  helper.add_lookup_species("H2O");
  helper.add_lookup_species("N2");
  helper.add_lookup_species("mixture_molecular_weight");

}


//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
CharOxidationShaddix::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CharOxidationShaddix::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidationShaddix::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_particletempLabel);
  tsk->computes(d_surfacerateLabel);
  tsk->computes(d_PO2surfLabel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidationShaddix::initVars( const ProcessorGroup * pc,
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


  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
CharOxidationShaddix::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  // get gas phase temperature label
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }
  // get gas phase O2 label
  if (VarLabel::find("O2")) {
    _O2_varlabel = VarLabel::find("O2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase O2.",__FILE__,__LINE__);
  }
  // get gas phase CO2 label
  if (VarLabel::find("CO2")) {
    _CO2_varlabel = VarLabel::find("CO2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase CO2.",__FILE__,__LINE__);
  }
  // get gas phase H2O label
  if (VarLabel::find("H2O")) {
    _H2O_varlabel = VarLabel::find("H2O");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase H2O.",__FILE__,__LINE__);
  }
  // get gas phase N2 label
  if (VarLabel::find("N2")) {
    _N2_varlabel = VarLabel::find("N2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase N2.",__FILE__,__LINE__);
  }
  // get gas phase mixture_molecular_weight label
  if (VarLabel::find("mixture_molecular_weight")) {
    _MW_varlabel = VarLabel::find("mixture_molecular_weight");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase mixture_molecular_weight.",__FILE__,__LINE__);
  }

  std::string taskname = "CharOxidationShaddix::sched_computeModel";
  Task* tsk = scinew Task(taskname, this, &CharOxidationShaddix::computeModel, timeSubStep );

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_particletempLabel);
    tsk->computes(d_surfacerateLabel);
    tsk->computes(d_PO2surfLabel);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_particletempLabel);
    tsk->modifies(d_surfacerateLabel);
    tsk->modifies(d_PO2surfLabel);
    which_dw = Task::NewDW;
  }

  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 );
  tsk->requires( which_dw, _number_density_varlabel, gn, 0 );
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 );
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _charmass_weighted_scaled_varlabel, gn, 0 );
  tsk->requires( which_dw, _rcmass_weighted_scaled_varlabel, gn, 0 );

  for (int i=0; i<_nQn_part;i++ ){
  tsk->requires( which_dw, _length_varlabel[i], gn, 0 );
  tsk->requires( which_dw, _weight_varlabel[i], gn, 0 );
  }

  tsk->requires( which_dw, _gas_temperature_varlabel, gn, 0);
  tsk->requires( which_dw, _O2_varlabel, gn, 0 );
  tsk->requires( which_dw, _CO2_varlabel, gn, 0 );
  tsk->requires( which_dw, _H2O_varlabel, gn, 0 );
  tsk->requires( which_dw, _N2_varlabel, gn, 0 );
  tsk->requires( which_dw, _MW_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label());
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, _RC_RHS_source_varlabel, gn, 0 );

  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0);
  tsk->requires( Task::NewDW, _devolCharLabel, gn, 0);
  tsk->requires( Task::NewDW, _devolRCLabel, gn, 0);
  if ( _char_birth_label != nullptr )
    tsk->requires( Task::NewDW, _char_birth_label, gn, 0 );
  if ( _rawcoal_birth_label != nullptr )
    tsk->requires( Task::NewDW, _rawcoal_birth_label, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CharOxidationShaddix::computeModel( const ProcessorGroup * pc,
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
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( char_rate, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_char_rate, d_gasLabel, matlIndex, patch );
      new_dw->getModifiable( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      new_dw->getModifiable( surface_rate, d_surfacerateLabel, matlIndex, patch );
      new_dw->getModifiable( PO2surf_, d_PO2surfLabel, matlIndex, patch );
    }

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
    constCCVariable<double> rawcoal_weighted_scaled;
    which_dw->get( rawcoal_weighted_scaled, _rcmass_weighted_scaled_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> char_weighted_scaled;
    which_dw->get( char_weighted_scaled, _charmass_weighted_scaled_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> RHS_source;
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RC_RHS_source;
    new_dw->get( RC_RHS_source , _RC_RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> number_density;
    which_dw->get( number_density , _number_density_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> O2;
    which_dw->get( O2, _O2_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> CO2;
    which_dw->get( CO2, _CO2_varlabel, matlIndex, patch, gn, 0 );
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
    if ( _rawcoal_birth_label != nullptr ){
      add_rawcoal_birth = true;
      new_dw->get( rawcoal_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 );
    }
    if ( _char_birth_label != nullptr ){
      add_char_birth = true;
      new_dw->get( char_birth, _rawcoal_birth_label, matlIndex, patch, gn, 0 );
    }



  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

      Uintah::parallel_for( range, [&](int i, int j, int k) {
         double max_char_reaction_rate_O2_;
         double char_reaction_rate_;
         double char_production_rate_;
         double particle_temp_rate_;
         int NIter;
         double rc_destruction_rate_;
         double PO2_surf=0.0;
         double PO2_surf_guess;
         double PO2_surf_tmp;
         double PO2_surf_new;
         double PO2_surf_old;
         double CO2CO;
         double OF;
         double ks;
         double q;

         double d_tol;
         double delta;
         double Conc;
         double DO2;
         double gamma;
         double f0;
         double f1;
         int icount;

         if (weight[d_quadNode](i,j,k)/_weight_scaling_constant < _weight_small) {
           char_production_rate_ = 0.0;
           char_rate(i,j,k) = 0.0;
           gas_char_rate(i,j,k) = 0.0;
           particle_temp_rate(i,j,k) = 0.0;
           surface_rate(i,j,k) = 0.0;

         } else {
           double denph=den(i,j,k);
           double temperatureph=temperature(i,j,k);
           double particle_temperatureph=particle_temperature(i,j,k);
           double lengthph=length[d_quadNode](i,j,k);
           double rawcoal_massph=rawcoal_mass(i,j,k);
           double char_massph=char_mass(i,j,k);
           double weightph=weight[d_quadNode](i,j,k);
           double O2ph=O2(i,j,k);
           double CO2ph=CO2(i,j,k);
           double H2Oph=H2O(i,j,k);
           double N2ph=N2(i,j,k);
           double MWmixph=MWmix(i,j,k);
           double devolCharph=devolChar(i,j,k);
           double devolRCph=devolRC(i,j,k);
           double RHS_sourceph=RHS_source(i,j,k);

           double PO2_inf = O2ph/_WO2/MWmixph;
           double AreaSum =0;
           for (int ix=0; ix<_nQn_part;ix++ ){ 
             AreaSum+=  weight[ix](i,j,k)*length[ix](i,j,k)*length[ix](i,j,k);
           }
           double surfaceAreaFraction=weightph*lengthph*lengthph/AreaSum;
        


           if((PO2_inf < 1e-12) || (rawcoal_massph+char_massph) < _small) {
             PO2_surf = 0.0;
             CO2CO = 0.0;
             q = 0.0;
           } else {
             char_reaction_rate_ = 0.0;
             char_production_rate_ = 0.0;
             particle_temp_rate_ = 0.0;
             NIter = 61;
             delta = PO2_inf/4.0;
             d_tol = 1e-15;
             PO2_surf_old = 0.0;
             PO2_surf_new = 0.0;
             PO2_surf_tmp = 0.0;
             PO2_surf_guess = 0.0;
             f0 = 0.0;
             f1 = 0.0;
             PO2_surf = 0.0;
             CO2CO = 0.0;
             q = 0.0;
             icount = 0;
          
          // Calculate O2 diffusion coefficient
             DO2 = (CO2ph/_WCO2 + H2Oph/_WH2O + N2ph/_WN2)/(CO2ph/(_WCO2*_D1) + 
                   H2Oph/(_WH2O*_D2) + 
                   N2ph/(_WN2*_D3))*(std::pow((temperatureph/_T0),1.5));
          // Concentration C = P/RT
             Conc = MWmixph*denph*1000.0;
             ks = _As*exp(-_Es/(_R*particle_temperatureph));

             double   expFactor=std::exp(3070.0/particle_temperatureph);
             PO2_surf_guess = PO2_inf/2.0;
             PO2_surf_old = PO2_surf_guess-delta;
             CO2CO = 0.02*(std::pow(PO2_surf_old,0.21))*expFactor;
             OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
             gamma = -(1.0-OF);
             q = ks*(std::pow(PO2_surf_old,_n));
             f0 = PO2_surf_old - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));

             PO2_surf_new = PO2_surf_guess+delta;
             CO2CO = 0.02*(std::pow(PO2_surf_new,0.21))*expFactor;
             OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
             gamma = -(1.0-OF);
             q = ks*(std::pow(PO2_surf_new,_n));
             f1 = PO2_surf_new - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));

             for ( int iter=0; iter < NIter; iter++) {
               icount++;
               PO2_surf_tmp = PO2_surf_old;
               PO2_surf_old=PO2_surf_new;
               PO2_surf_new=PO2_surf_tmp - (PO2_surf_new - PO2_surf_tmp)/(f1-f0) * f0;
               PO2_surf_new = std::max(0.0, std::min(PO2_inf, PO2_surf_new));            
               if (std::abs(PO2_surf_new-PO2_surf_old) < d_tol){
                 PO2_surf=PO2_surf_new;
                 CO2CO = 0.02*(std::pow(PO2_surf,0.21))*expFactor;
                 OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                 gamma = -(1.0-OF);
                 q = ks*(std::pow(PO2_surf,_n));
                 break;
               }
               f0 = f1;
               CO2CO = 0.02*(std::pow(PO2_surf_new,0.21))*expFactor;
               OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
               gamma = -(1.0-OF);
               q = ks*(std::pow(PO2_surf_new,_n));
               f1 = PO2_surf_new - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
               PO2_surf=PO2_surf_new; // This is needed to assign PO2_surf if we don't converge.
             }
           }

           char_production_rate_ = devolCharph;
           rc_destruction_rate_ = devolRCph;
           double gamma1=(_WC/_WO2)*((CO2CO+1.0)/(CO2CO+0.5)); 
           max_char_reaction_rate_O2_ = std::max( (O2ph*denph*gamma1*surfaceAreaFraction)/(dt*weightph) , 0.0 );

           double max_char_reaction_rate_ = 0.0;

           if ( add_rawcoal_birth && add_char_birth ){ 
             max_char_reaction_rate_ = std::max((rawcoal_massph+char_massph)/(dt) 
                    +( (RHS_sourceph + RC_RHS_source(i,j,k)) / (vol*weightph) + (char_production_rate_ + rc_destruction_rate_
                        +   char_birth(i,j,k) + rawcoal_birth(i,j,k) )/ weightph )
                    *_char_scaling_constant*_weight_scaling_constant, 0.0); // equation assumes RC_scaling=Char_scaling
           } else { 
             max_char_reaction_rate_ = std::max((rawcoal_massph+char_massph)/(dt) 
                  +((RHS_sourceph + RC_RHS_source(i,j,k)) / (vol*weightph) + (char_production_rate_ + rc_destruction_rate_
                      )/ weightph )
                  *_char_scaling_constant*_weight_scaling_constant, 0.0); // equation assumes RC_scaling=Char_scaling
           }


           max_char_reaction_rate_ = std::min( max_char_reaction_rate_ ,max_char_reaction_rate_O2_ );
           char_reaction_rate_ = std::min(_pi*(std::pow(lengthph,2.0))*_WC*q , max_char_reaction_rate_); // kg/(s.#)    

           particle_temp_rate_ = -char_reaction_rate_/_WC/(1.0+CO2CO)*(CO2CO*_HF_CO2 + _HF_CO); // J/(s.#)
           char_rate(i,j,k) = (-char_reaction_rate_*weightph+char_production_rate_)/(_char_scaling_constant*_weight_scaling_constant);
           gas_char_rate(i,j,k) = char_reaction_rate_*weightph;// kg/(m^3.s)
           particle_temp_rate(i,j,k) = particle_temp_rate_*weightph; // J/(s.m^3)
           surface_rate(i,j,k) = -_WC*q;  // in kg/s/m^2
           PO2surf_(i,j,k) = PO2_surf;
        //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
           if( char_rate(i,j,k)>0.0 ) {
             char_rate(i,j,k) = 0;
             gas_char_rate(i,j,k) = 0;
             particle_temp_rate(i,j,k) = 0;
             surface_rate(i,j,k) = 0;  // in kg/s/m^2
             PO2surf_(i,j,k) = PO2_surf;
           }
         }
       } );

  }//end patch loop
}
