#include <CCA/Components/Arches/CoalModels/ShrinkageRate.h>

#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

#include <boost/math/special_functions/erf.hpp>
//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
ShrinkageRateBuilder::ShrinkageRateBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            MaterialManagerP          & materialManager,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
}

ShrinkageRateBuilder::~ShrinkageRateBuilder(){}

ModelBase* ShrinkageRateBuilder::build() {
  return scinew ShrinkageRate( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

ShrinkageRate::ShrinkageRate( std::string modelName,
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
  //constants
  // _pi = acos(-1.0);
  // _C_tm = 0.461; // [=] m/s/Kelvin
  // _C_t = 3.32; // [=] dimensionless
  // _C_m = 1.19; // [=] dimensionless
  // _Adep = 2.4; // [=] dimensionless
}

ShrinkageRate::~ShrinkageRate()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
ShrinkageRate::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

  // Get size scaling constant
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  std::string length_name = ArchesCore::append_env( length_root, d_quadNode );
  std::string lengthqn_name = ArchesCore::append_qn_env( length_root, d_quadNode );
  EqnBase& temp_length_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(lengthqn_name);
  DQMOMEqn& length_eqn = dynamic_cast<DQMOMEqn&>(temp_length_eqn);
  m_scaling_const_length = length_eqn.getScalingConstant(d_quadNode);


  // get weight scaling constant
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode);
  m_weight_scaled_varlabel = VarLabel::find(weightqn_name);

  // Get rates from char oxidation model
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  CharOxiModelMap charoximodels_ = modelFactory.retrieve_charoxi_models();
  for( CharOxiModelMap::iterator iModel = charoximodels_.begin(); iModel != charoximodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      m_charoxiSize_varlabel = iModel->second->getParticleSizeSourceLabel();
    }
  }

  // make sure the right char oxidation model is used
  std::string modelName;
  bool tflag = true;
  ProblemSpecP db_coal_models = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Models");
  for ( ProblemSpecP db_model = db_coal_models->findBlock("model"); db_model != nullptr; db_model = db_model->findNextBlock("model")){
    db_model->getAttribute("type", modelName);
    if( modelName == "CharOxidationSmith2016" ) {
      tflag = false;
    }
  }
  if (tflag) {
    throw ProblemSetupException("Error: CharOxidationSmith2016 is required for ShrinkageRate block in <DQMOM-> Models>.", __FILE__, __LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
ShrinkageRate::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ShrinkageRate::initVars";
  Task* tsk = scinew Task(taskname, this, &ShrinkageRate::initVars);

  tsk->computes(d_modelLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
ShrinkageRate::initVars( const ProcessorGroup * pc,
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

    CCVariable<double> shr_rate;

    new_dw->allocateAndPut( shr_rate, d_modelLabel, matlIndex, patch );
    shr_rate.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
ShrinkageRate::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ShrinkageRate::computeModel";
  Task* tsk = scinew Task(taskname, this, &ShrinkageRate::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;
  // Ghost::GhostType  gac = Ghost::AroundCells;

  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    which_dw = Task::NewDW;
  }

  //tsk->requires( Task::NewDW, _surfacerate_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, m_charoxiSize_varlabel, gn, 0 ); 
  tsk->requires( which_dw, m_weight_scaled_varlabel, gn, 0 ); 
  
  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
ShrinkageRate::computeModel( const ProcessorGroup * pc,
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
 
    CCVariable<double> shr_rate;
    //Task::WhichDW which_dw;
    DataWarehouse* which_dw;
    
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( shr_rate, d_modelLabel, matlIndex, patch );
      shr_rate.initialize(0.0);
      //which_dw = Task::OldDW;
      which_dw = old_dw;
    } else {
      new_dw->getModifiable( shr_rate, d_modelLabel, matlIndex, patch );
      //which_dw = Task::NewDW;
      which_dw = new_dw;
    }
    constCCVariable<double> size_rate;
    new_dw->get(size_rate, m_charoxiSize_varlabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> weight;
    which_dw->get(weight, m_weight_scaled_varlabel, matlIndex, patch, gn, 0 );

    // Parallel Loop
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for( range, [&](int i, int j, int k) {
    	//shr_rate(i,j,k) = size_rate(i,j,k)*weight(i,j,k)/m_scaling_const_length;
    	shr_rate(i,j,k) = size_rate(i,j,k);
    });
    
  }//end patch loop
}

