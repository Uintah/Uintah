#include <CCA/Components/Arches/CoalModels/LinearSwelling.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

#include <functional>

//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
LinearSwellingBuilder::LinearSwellingBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            MaterialManagerP          & materialManager,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
}

LinearSwellingBuilder::~LinearSwellingBuilder(){}

ModelBase* LinearSwellingBuilder::build() {
  return scinew LinearSwelling( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

LinearSwelling::LinearSwelling( std::string modelName,
                                              MaterialManagerP& materialManager,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames,
                                              vector<std::string> scalarLabelNames,
                                              int qn )
: ModelBase(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );
  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  //initialize
  m_birth_label = nullptr;
}

LinearSwelling::~LinearSwelling()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
LinearSwelling::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();
  CoalHelper& coal_helper = CoalHelper::self();

  // get devol source term label and devol label from the devolatilization model
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == qn) {
      m_devolRCLabel = iModel->second->getModelLabel() ;
    }
  }

  m_init_diam = ArchesCore::get_inlet_particle_size( db, qn );
  double init_particle_density = ArchesCore::get_inlet_particle_density( db );
  double ash_mass_frac = coal_helper.get_coal_db().ash_mf;
  double p_volume = M_PI/6.*m_init_diam*m_init_diam*m_init_diam; // particle volme [m^3]
  m_init_rc = p_volume*init_particle_density*(1.-ash_mass_frac);
  m_minimum_particle_diameter = coal_helper.get_coal_db().min_particle_size[qn]; //min particle size when particle is ash 

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // Get length scaling constant
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  std::string length_name = ArchesCore::append_env( length_root, qn );
  std::string lengthqn_name = ArchesCore::append_qn_env( length_root, qn );
  EqnBase& temp_length_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(lengthqn_name);
  DQMOMEqn& length_eqn = dynamic_cast<DQMOMEqn&>(temp_length_eqn);
  m_scaling_const_length = length_eqn.getScalingConstant(qn);
  std::string lengthqn_RHS = lengthqn_name+"_RHS";
  m_RHS_source_varlabel = VarLabel::find(lengthqn_RHS);
  m_weighted_length_label = VarLabel::find(lengthqn_name);
  //get the birth term if any:
  const std::string birth_name = length_eqn.get_model_by_type( "BirthDeath" );
  std::string birth_qn_name = ArchesCore::append_qn_env(birth_name, d_quadNode);
  if ( birth_name != "NULLSTRING" ){
    m_birth_label = VarLabel::find( birth_qn_name );
  }

  // Need weight name and scaling constant
  std::string weight_name = ArchesCore::append_qn_env("w", qn);
  std::string weight_RHS_name = weight_name + "_RHS";
  m_RHS_weight_varlabel = VarLabel::find(weight_RHS_name);
  std::string scaled_weight_name = ArchesCore::append_qn_env("w", d_quadNode);
  m_scaled_weight_varlabel = VarLabel::find(scaled_weight_name);
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode);

  // Get rcmass scaling constant
  std::string rc_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
  std::string rc_name = ArchesCore::append_env( rc_root, qn );
  std::string rcqn_name = ArchesCore::append_qn_env( rc_root, qn );
  EqnBase& temp_rc_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcqn_name);
  DQMOMEqn& rc_eqn = dynamic_cast<DQMOMEqn&>(temp_rc_eqn);
  m_scaling_const_rc = rc_eqn.getScalingConstant(qn);

  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

  if (db_coal_props->findBlock("LinearSwelling")) {
    ProblemSpecP db_LS = db_coal_props->findBlock("LinearSwelling");
    db_LS->getWithDefault("Fsw",m_Fsw,0.05); //swelling factor
  } else {
    throw ProblemSetupException("Error: LinearSwelling block missing in <ParticleProperties>.", __FILE__, __LINE__);
  }

  if (db_coal_props->findBlock("FOWYDevol")) {
    ProblemSpecP db_BT = db_coal_props->findBlock("FOWYDevol");
    db_BT->require("v_hiT", m_v_hiT); // this is a
  } else {
    throw ProblemSetupException("Error: LinearSwelling Model requires FOWYDevol model block in <ParticleProperties>.", __FILE__, __LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
LinearSwelling::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "LinearSwelling::initVars";
  Task* tsk = scinew Task(taskname, this, &LinearSwelling::initVars);

  tsk->computes(d_modelLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
LinearSwelling::initVars( const ProcessorGroup * pc,
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

    CCVariable<double> ls_rate;

    new_dw->allocateAndPut( ls_rate, d_modelLabel, matlIndex, patch );
    ls_rate.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
LinearSwelling::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "LinearSwelling::computeModel";
  Task* tsk = scinew Task(taskname, this, &LinearSwelling::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;
  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    which_dw = Task::NewDW;
  }
  tsk->requires( Task::NewDW, m_devolRCLabel, gn, 0 );
  tsk->requires( Task::NewDW, m_RHS_source_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, m_RHS_weight_varlabel, gn, 0 );
  tsk->requires( which_dw, m_scaled_weight_varlabel, gn, 0 );
  tsk->requires( which_dw, m_weighted_length_label, gn, 0 );
  if ( m_birth_label != nullptr )
    tsk->requires( Task::NewDW, m_birth_label, gn, 0 );

  // get time step size for model clipping
  tsk->requires( Task::OldDW,d_fieldLabels->d_delTLabel, Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
LinearSwelling::computeModel( const ProcessorGroup * pc,
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
    const double vol = Dx.x()* Dx.y()* Dx.z();

    CCVariable<double> ls_rate;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( ls_rate, d_modelLabel, matlIndex, patch );
      ls_rate.initialize(0.0);
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( ls_rate, d_modelLabel, matlIndex, patch );
      ls_rate.initialize(0.0);
    }
    constCCVariable<double> weighted_length;
    which_dw->get( weighted_length, m_weighted_length_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> devol_rate;
    new_dw->get(devol_rate, m_devolRCLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> scaled_weight;
    which_dw->get( scaled_weight, m_scaled_weight_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> RHS_source;
    new_dw->get( RHS_source , m_RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RHS_weight;
    new_dw->get( RHS_weight , m_RHS_weight_varlabel , matlIndex , patch , gn , 0 );
    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    constCCVariable<double> birth;
    bool add_birth = false;
    if ( m_birth_label != nullptr ){
      new_dw->get( birth, m_birth_label, matlIndex, patch, gn, 0 );
      add_birth = true;
    }
    const double dt = DT;
    std::function<double  ( int i,  int j, int k)> lambdaBirth;
    if ( add_birth ){
      lambdaBirth = [&]( int i, int j, int k)-> double   { return  birth(i,j,k);};
    }else{
      lambdaBirth = [&]( int i, int j, int k)-> double   { return 0.0;};
    }

    // if Fsw is bigger than 1 diameter of particle will increase, and maximum diameter will be m_Fsw*m_init_diam
    double sign = 1. ;
    double min_diameter = m_Fsw*m_init_diam;
    if (m_Fsw < 1) {
      // if Fsw is lower than 1 diamter of particle will decrease, but char oxidation model is also decreasing size, thus, min diameter is when the particle is only ash.
      sign = -1.;
      min_diameter = m_minimum_particle_diameter; 
    }

    double rate = 0.0;
    double max_rate = 0.0;
    double updated_weight = 0.0;
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for( range, [&](int i, int j, int k) {
      rate = - m_init_diam * (m_Fsw-1.0) * devol_rate(i,j,k) * m_scaling_const_rc / m_v_hiT / m_init_rc / m_scaling_const_length; // [1/m^2/s] - negative sign makes this source-term positive if m_Fsw is bigger than 1, otherwise source-term is negavite and particle will decrease in size 
      // particle size won't increase above m_Fsw*m_init_diam
      updated_weight = std::max(scaled_weight(i,j,k) + dt / vol * ( RHS_weight(i,j,k) ) , 1e-15);
      max_rate = (updated_weight * min_diameter / m_scaling_const_length - weighted_length(i,j,k) ) / dt - ( RHS_source(i,j,k) / vol + lambdaBirth(i,j,k));
      ls_rate(i,j,k) = std::min(sign*rate,sign*max_rate); // if Fsw < 1 m_sign is -
      ls_rate(i,j,k) *= sign; // add sign back to source  
    });

  }//end patch loop
}
