#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, 
                      SimulationStateP& sharedState,
                      ArchesLabel* fieldLabels,
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

// Constructor/destructor for parent class is also called from constructor/destructor
// of child class.
// 
// Functions defined here will be overridden if redefined in a child class.
// The child class can explicitly call the parent class method, like this:
// ModelBase::some_function();
//
// Functions declared as pure virtual functions MUST be redefined in child class.
//
// Functions not redefined in child class will use the ModelBase version.


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
/** @details  
This method is the same for all models, as all models must require() and compute()
 the gas phase source term and the actual model term.

However, the implementation (as opposed to the schedule) requires knowledge of the 
 variable type, so the dummyInit() method must be defined explicitly in the 
 particular model classes - not here.
 */
/*
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
*/
