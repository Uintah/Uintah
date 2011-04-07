#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
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
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

CharOxidation::CharOxidation( std::string modelName, 
                              SimulationStateP& sharedState,
                              ArchesLabel* fieldLabels,
                              vector<std::string> icLabelNames, 
                              vector<std::string> scalarLabelNames,
                              int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_quadNode = qn;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  R_   =  1.987;       // [=] kcal/kmol : ideal gas constant
  pi_  = 3.1415926535;
}

CharOxidation::~CharOxidation()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
CharOxidation::problemSetup(const ProblemSpecP& params)
{
  // This method is called by all char oxiation classes' problemSetup()

  ProblemSpecP db = params; 

  // set model clipping (not used yet...)
  db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  // grab weight scaling factor and small value
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);

  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_constant = weight_eqn.getScalingConstant();
  d_weight_label = weight_eqn.getTransportEqnLabel();
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
CharOxidation::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CharOxidation::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &CharOxidation::dummyInit);

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
CharOxidation::dummyInit( const ProcessorGroup* pc,
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
CharOxidation::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CharOxidation::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidation::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidation::initVars( const ProcessorGroup * pc, 
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
    model_value.initialize(0.0);

    CCVariable<double> gas_value; 
    new_dw->allocateAndPut( gas_value, d_gasLabel, matlIndex, patch ); 
    gas_value.initialize(0.0);
  }
}

