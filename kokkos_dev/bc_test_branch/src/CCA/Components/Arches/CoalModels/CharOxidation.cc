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

  // Create the particle temperature source term associated with this model
  std::string particletempSourceName = modelName + "_particletempSource";
  d_particletempLabel = VarLabel::create( particletempSourceName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_particletempLabel);

  // Create the char oxidation surface rate term associated with this model
  std::string surfacerateName = modelName + "_surfacerate";
  d_surfacerateLabel = VarLabel::create( surfacerateName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_surfacerateLabel);

  // Create the char oxidation PO2 surf term associated with this model
  std::string PO2surfName = modelName + "_PO2surf";
  d_PO2surfLabel = VarLabel::create( PO2surfName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_PO2surfLabel);

}

CharOxidation::~CharOxidation()
{
  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
       iter != _extra_local_labels.end(); iter++) {
    VarLabel::destroy( *iter );
  }
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
CharOxidation::problemSetup(const ProblemSpecP& params, int qn)
{
  // This method is called by all devolatilizaiton classes' problemSetup()

  ProblemSpecP db = params; 

  // set model clipping (not used yet...)
  db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  db->getWithDefault( "high_clip", d_highModelClip, 999999.0 );

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
}

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

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
       iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
    tsk->requires( Task::OldDW, *iter,   gn, 0);
  }

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

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
         iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      constCCVariable<double> oldtempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
      old_dw->get( oldtempVar, *iter,   matlIndex, patch, gn, 0 );
      tempVar.copyData(oldtempVar);
    }

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
/** @details
This method intentionally does nothing.

@see CharOxidation::initVars()
*/
void 
CharOxidation::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CharOxidation::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidation::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
/** @details
This method is left intentionally blank.  This way, if the method is
 called for a heat transfer model, and that heat transfer model
 doesn't require the initialization of any variables, the child 
 class will not need to re-define this (empty) method.

If additional variables are needed, and initVars needs to do stuff,
 the model can redefine it.
*/
void
CharOxidation::initVars( const ProcessorGroup * pc, 
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

  }
  */
  
}
