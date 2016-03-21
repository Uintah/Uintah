#include <CCA/Components/Arches/CoalModels/ParticleConvection.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
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

ParticleConvectionBuilder::ParticleConvectionBuilder( const std::string         & modelName,
                                        const vector<std::string> & reqICLabelNames,
                                        const vector<std::string> & reqScalarLabelNames,
                                        ArchesLabel         * fieldLabels,
                                        SimulationStateP          & sharedState,
                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

ParticleConvectionBuilder::~ParticleConvectionBuilder(){}

ModelBase* ParticleConvectionBuilder::build(){
  return new ParticleConvection( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

ParticleConvection::ParticleConvection( std::string           modelName,
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

ParticleConvection::~ParticleConvection()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
ParticleConvection::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb;


}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
ParticleConvection::sched_initVars( const LevelP& level, SchedulerP& sched )
{
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
ParticleConvection::initVars( const ProcessorGroup * pc,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw )
{
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model
//---------------------------------------------------------------------------
void
ParticleConvection::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ParticleConvection::computeModel";
  Task* tsk = new Task(taskname, this, &ParticleConvection::computeModel, timeSubStep );

  if (d_timeSubStep == 0) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
  }

  tsk->requires(Task::OldDW, VarLabel::find("volFraction"), Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
ParticleConvection::computeModel( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> model;
    CCVariable<double> gas_source;

    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
      model.initialize(0.0);
    } else {
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch );
    }

    constCCVariable<double> vol_fraction;

    old_dw->get( vol_fraction, VarLabel::find("volFraction"), matlIndex, patch, Ghost::None, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      if ( vol_fraction[c] > 0. ){


      } else {

        model[c] = 0.0;

      }

      gas_source[c] = 0.0;

    }
  }
}
