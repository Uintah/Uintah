#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, 
                      SimulationStateP& sharedState,
                      const ArchesLabel* fieldLabels,
                      vector<std::string> reqICLabelNames, 
                      vector<std::string> reqScalarLabelNames,
                      int qn ) : 
            d_modelName(modelName),  d_sharedState( sharedState ), d_fieldLabels(fieldLabels), 
            d_icLabels(reqICLabelNames), d_scalarLabels(reqScalarLabelNames), d_quadNode(qn)
{
  // The type and number of d_modelLabel and d_gasLabel
  // is model-dependent, so the creation of these labels 
  // go in the model class constructor.
  // (Note that the labels themselves are still defined in 
  //  the parent class...)
 
  d_labelSchedInit  = false; 
}

ModelBase::~ModelBase()
{
  VarLabel::destroy(d_modelLabel); 
  VarLabel::destroy(d_gasLabel); 
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
ModelBase::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ModelBase::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ModelBase::dummyInit);

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}
void 
ModelBase::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )		      
{
  // As before, the variable type of d_modelLabel and d_gasLabel are
  // model-dependent, so the new_dw->allocateAndPut/old_dw->get statements
  // must be in specific models' ModelName::initVars() method.
}

