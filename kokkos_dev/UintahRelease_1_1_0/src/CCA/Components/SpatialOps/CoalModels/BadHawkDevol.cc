#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/SpatialOps/CoalModels/BadHawkDevol.h>


//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
BadHawkDevolBuilder::BadHawkDevolBuilder( const std::string         & modelName, 
                                          const vector<std::string> & icLabelNames, 
                                          const Fields              * fieldLabels,
                                          SimulationStateP          & sharedState,
                                          int qn) :
  ModelBuilder( modelName, fieldLabels, icLabelNames, sharedState, qn )
{}

BadHawkDevolBuilder::~BadHawkDevolBuilder(){}

ModelBase*
BadHawkDevolBuilder::build(){
  return scinew BadHawkDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, 
                              d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

BadHawkDevol::BadHawkDevol( std::string srcName, SimulationStateP& sharedState,
                            const Fields* fieldLabels,
                            vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, fieldLabels, icLabelNames, qn)
{}

BadHawkDevol::~BadHawkDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
BadHawkDevol::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb; 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
BadHawkDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BadHawkDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &BadHawkDevol::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
  } else {
    tsk->modifies(d_modelLabel); 
  }

  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE MODEL
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BadHawkDevol::computeModel( const ProcessorGroup* pc, 
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
    int matlIndex = 0;

    CCVariable<double> model; 
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
    }

    for (vector<std::string>::iterator iter = d_icLabels.begin(); 
         iter != d_icLabels.end(); iter++) { 
    
    }

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      //model[c] =
    }
  }
}
