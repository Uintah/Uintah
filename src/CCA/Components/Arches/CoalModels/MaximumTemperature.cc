#include <CCA/Components/Arches/CoalModels/MaximumTemperature.h>
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
MaximumTemperatureBuilder::MaximumTemperatureBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            MaterialManagerP          & materialManager,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
}

MaximumTemperatureBuilder::~MaximumTemperatureBuilder(){}

ModelBase* MaximumTemperatureBuilder::build() {
  return scinew MaximumTemperature( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

MaximumTemperature::MaximumTemperature( std::string modelName, 
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
}

MaximumTemperature::~MaximumTemperature()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
MaximumTemperature::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
  
  // create max T var label and get scaling constant
  std::string max_pT_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_MAXTEMPERATURE); 
  std::string max_pT_name = ArchesCore::append_env( max_pT_root, d_quadNode ); 
  std::string max_pTqn_name = ArchesCore::append_qn_env( max_pT_root, d_quadNode ); 
  _max_pT_varlabel = VarLabel::find(max_pT_name);
  _max_pT_weighted_scaled_varlabel = VarLabel::find(max_pTqn_name); 
  EqnBase& temp_max_pT_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(max_pTqn_name);
  DQMOMEqn& max_pT_eqn = dynamic_cast<DQMOMEqn&>(temp_max_pT_eqn);
   _max_pT_scaling_constant = max_pT_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = max_pTqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  
  // create particle temperature label
  std::string temperature_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE); 
  std::string temperature_name = ArchesCore::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
 
  // get weight scaling constant
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode); 
  std::string weight_name = ArchesCore::append_env("w", d_quadNode); 
  _weight_scaled_varlabel = VarLabel::find(weightqn_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
MaximumTemperature::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MaximumTemperature::initVars"; 
  Task* tsk = scinew Task(taskname, this, &MaximumTemperature::initVars);

  tsk->computes(d_modelLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
MaximumTemperature::initVars( const ProcessorGroup * pc, 
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

    CCVariable<double> maxT_rate;
    
    new_dw->allocateAndPut( maxT_rate, d_modelLabel, matlIndex, patch );
    maxT_rate.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
MaximumTemperature::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "MaximumTemperature::computeModel";
  Task* tsk = scinew Task(taskname, this, &MaximumTemperature::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel); 
    which_dw = Task::NewDW; 
  }
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _max_pT_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _weight_scaled_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _max_pT_weighted_scaled_varlabel, gn, 0 ); 
  tsk->requires( Task::OldDW, d_fieldLabels->d_delTLabel); 
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 ); 

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" )); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
MaximumTemperature::computeModel( const ProcessorGroup * pc, 
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

    CCVariable<double> maxT_rate;
    DataWarehouse* which_dw; 

    if ( timeSubStep == 0 ){ 
      which_dw = old_dw; 
      new_dw->allocateAndPut( maxT_rate, d_modelLabel, matlIndex, patch );
      maxT_rate.initialize(0.0);
    } else { 
      which_dw = new_dw; 
      new_dw->getModifiable( maxT_rate, d_modelLabel, matlIndex, patch ); 
    }

    constCCVariable<double> pT; 
    which_dw->get( pT , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> max_pT; 
    which_dw->get( max_pT    , _max_pT_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> scaled_weight; 
    which_dw->get( scaled_weight , _weight_scaled_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RHS_source; 
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> max_pTqn; 
    which_dw->get( max_pTqn, _max_pT_weighted_scaled_varlabel, matlIndex , patch , gn , 0 );


    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      maxT_rate[c] = 0;
      if (pT[c] > max_pT[c]) {
        // compute rate such that max_pT = pT
        maxT_rate[c]=scaled_weight[c]/_max_pT_scaling_constant*(pT[c]-max_pT[c])/dt - RHS_source[c]/vol;
      }

    }//end cell loop
  }//end patch loop
}
