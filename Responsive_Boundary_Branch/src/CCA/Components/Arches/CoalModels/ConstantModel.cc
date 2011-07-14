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
}

ConstantModel::~ConstantModel()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ConstantModel::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb; 

  db->require("constant",d_constant); 

}

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
  // This method left intentionally blank...
  // It has the form:
  /*
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> something; 
    new_dw->allocateAndPut( something, d_something_label, matlIndex, patch ); 
    something.initialize(0.0)

  }
  */
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model 
//---------------------------------------------------------------------------
void 
ConstantModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ConstantModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantModel::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel); 
  }

  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    // require any internal coordinates needed to compute the model
  }

  for( vector<string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter ) {
    // require any scalars needed to compute the model
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
                   DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model; 
    CCVariable<double> gas_source;

    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
    }

    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
    }


    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      //if(d_quadNode ==1){
      //  model[c] = -d_constant;
      //} else { 
      //  model[c] = d_constant;
      //}
      model[c] = d_constant;
      gas_source[c] = 0.0;
    }
  }
}
