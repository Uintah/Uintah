#include <CCA/Components/Arches/CoalModels/ConstantModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:

ConstantModelBuilder::ConstantModelBuilder( const std::string         & modelName, 
                                            const vector<std::string> & reqICLabelNames,
                                            const vector<std::string> & reqScalarLabelNames,
                                            ArchesLabel         * fieldLabels,
                                            SimulationStateP          & sharedState,
                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

ConstantModelBuilder::~ConstantModelBuilder(){}

ModelBase* ConstantModelBuilder::build(){
  return scinew ConstantModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

ConstantModel::ConstantModel( std::string           modelName, 
                              SimulationStateP    & sharedState,
                              ArchesLabel   * fieldLabels,
                              vector<std::string>   icLabelNames, 
                              vector<std::string>   scalarLabelNames,
                              int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  d_reachedLowClip = false;
}

ConstantModel::~ConstantModel()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ConstantModel::problemSetup(const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  db->require("constant",d_constant); 

  db->getWithDefault("gas_source",d_useGasSource,false);

  if( d_icLabels.size() != 1 ) {
    std::stringstream errmsg;
    errmsg << "ERROR: Arches: ConstantModel: You did not specify the correct number of internal coordinates in the <ICVars> block. ";
    errmsg << "You specified " << d_icLabels.size() << ", ConstantModel was expecting 1 dependent internal coordinate.\n";
    errmsg << "The \"label\" attribute of the internal coordinate variable's block must be the label of the internal coordinate. The \"role\" attribute does not matter.";
    throw ProblemSetupException(errmsg.str(),__FILE__,__LINE__);
  }


  string temp_label_name;

  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();

  temp_label_name = d_icLabels[0];
  
  temp_label_name += "_qn";
  temp_label_name += node;

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();

  // get weight information
  string temp_weight_name = "w_qn";
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmomFactory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);

  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_constant = weight_eqn.getScalingConstant();
  d_weight_label = weight_eqn.getTransportEqnLabel();

  // get internal coordinate information
  if( dqmomFactory.find_scalar_eqn( temp_label_name ) ) {
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(&dqmomFactory.retrieve_scalar_eqn(temp_label_name) );

    d_ic_label = eqn->getTransportEqnLabel();
    d_ic_scaling_constant = eqn->getScalingConstant();

    eqn->addModel(d_modelLabel);

    d_doLowClip  = eqn->doLowClip();
    d_doHighClip = eqn->doHighClip();
    if(d_doLowClip) {
      d_low  = eqn->getLowClip();
    }
    if(d_doHighClip) {
      d_high = eqn->getHighClip();
    }
  } else {
    string errmsg = "ERROR: Arches: ConstantModel: Could not find internal coordinate "+temp_label_name+" as a registered equation in DQMOMEqnFactory.\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  d_constant /= d_ic_scaling_constant;

}


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
/** @details  
This method is a dummy initialization required by MPMArches. 
All models must be required() and computed() to copy them over 
without actually doing anything.  (Silly, isn't it?)
 */
void
ConstantModel::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ConstantModel::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ConstantModel::dummyInit);

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Actually do the dummy initialization
//-------------------------------------------------------------------------
/** @details
 This is called from ExplicitSolver::noSolve(), which skips the first timestep
 so that the initial conditions are correct.

This method was originally in ModelBase, but it requires creating CCVariables
 for the model and gas source terms, and the CCVariable type (double, Vector, &c.)
 is model-dependent.  Putting the method here eliminates if statements in 
 ModelBase and keeps the ModelBase class as generic as possible.

@see ExplicitSolver::noSolve()
 */
void
ConstantModel::dummyInit( const ProcessorGroup* pc,
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw )
{
  for( int p=0; p < patches->size(); ++p ) {

    Ghost::GhostType  gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> ModelTerm;
    CCVariable<double> GasModelTerm;
    
    constCCVariable<double> oldModelTerm;
    constCCVariable<double> oldGasModelTerm;

    new_dw->allocateAndPut( ModelTerm,    d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( GasModelTerm, d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModelTerm,    d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasModelTerm, d_gasLabel,   matlIndex, patch, gn, 0 );
    
    ModelTerm.copyData(oldModelTerm);
    GasModelTerm.copyData(oldGasModelTerm);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
ConstantModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ConstantModel::initVars";
  Task* tsk = scinew Task(taskname, this, &ConstantModel::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
ConstantModel::initVars( const ProcessorGroup * pc, 
                         const PatchSubset    * patches, 
                         const MaterialSubset * matls, 
                         DataWarehouse        * old_dw, 
                         DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model_value; 
    new_dw->allocateAndPut( model_value, d_modelLabel, matlIndex, patch ); 

    CCVariable<double> gas_value; 
    new_dw->allocateAndPut( gas_value, d_gasLabel, matlIndex, patch ); 

    model_value.initialize( d_constant );
    gas_value.initialize( 0.0 );
    //for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
    //  IntVector c = *iter;
    //  gas_value[c] = -(d_constant*d_ic_scaling_constant)*(weight[c]*d_w_scaling_constant);
    //}

  }
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model 
//---------------------------------------------------------------------------
void 
ConstantModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ConstantModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantModel::computeModel, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  if( timeSubStep == 0 && !d_labelSchedInit ) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {
    
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);

    tsk->requires(Task::OldDW, d_weight_label, gn, 0);
    tsk->requires(Task::OldDW, d_ic_label, gn, 0);

  } else {

    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel); 

    tsk->requires(Task::NewDW, d_weight_label, gn, 0);
    tsk->requires(Task::NewDW, d_ic_label, gn, 0);

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ConstantModel::computeModel( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw,
                             int timeSubStep )
{
  Ghost::GhostType gn = Ghost::None;

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model; 
    CCVariable<double> gas_source;
    constCCVariable<double> internal_coordinate;
    constCCVariable<double> weight;

    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );

      old_dw->get( internal_coordinate, d_ic_label, matlIndex, patch, gn, 0 );
      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    } else {

      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch ); 

      new_dw->get( internal_coordinate, d_ic_label, matlIndex, patch, gn, 0 );
      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    }

    model.initialize( d_constant );
    
    for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
      IntVector c = *iter;
      if( d_useGasSource ) {
        gas_source[c] = -(d_constant*d_ic_scaling_constant)*(weight[c]*d_w_scaling_constant);
      }
    }

    if( d_doLowClip ) { 
      for( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ) {
        IntVector c = *iter;
        double icval;
        if( d_unweighted ) {
          icval = internal_coordinate[c]*d_ic_scaling_constant;
        } else {
          icval = (internal_coordinate[c]/weight[c])*d_ic_scaling_constant;
        }
        if( icval <= (d_low+TINY) ) {
          model[c] = 0.0;
          gas_source[c] = 0.0;
        }
      }
    }
  
    if( d_doHighClip ) {
      for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
        IntVector c = *iter;
        double icval;
        if( d_unweighted ) {
          icval = internal_coordinate[c]*d_ic_scaling_constant;
        } else { 
          icval = (internal_coordinate[c]/weight[c])*d_ic_scaling_constant;
        }
        if( icval >= d_high ) {
          model[c] = 0.0;
          gas_source[c] = 0.0;
        }
      }
    }
  
  }
}
