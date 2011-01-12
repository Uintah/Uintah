#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
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

ParticleDensity::ParticleDensity( std::string modelName, 
                  SimulationStateP& sharedState,
                  const ArchesLabel* fieldLabels,
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

  DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self();
  numQuadNodes = eqn_factory.get_quad_nodes();

  std::string density_name = "rhop_qn";
  std::string qnode;
  std::stringstream out;
  out << qn;
  qnode = out.str();
  d_density_label = VarLabel::create( density_name+qnode, CCVariable<double>::getTypeDescription() );

  pi = 3.1415926535;

  d_constantLength = false; 
}

ParticleDensity::~ParticleDensity()
{
  VarLabel::destroy(d_density_label);
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ParticleDensity::problemSetup(const ProblemSpecP& params )
{
  // This method is called by problemSetup() in child classes

  ProblemSpecP db = params; 

  // grab weight scaling constant and small value
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;

  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  
  d_weight_label = weight_eqn.getTransportEqnLabel();
  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_constant = weight_eqn.getScalingConstant();

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
ParticleDensity::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ParticleDensity::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ParticleDensity::dummyInit);

  tsk->requires( Task::OldDW, d_density_label, gn, 0);
  tsk->requires( Task::OldDW, d_modelLabel,    gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,      gn, 0);

  tsk->computes(d_density_label);
  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  if( d_constantLength ) {
    tsk->computes(d_length_label);
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
ParticleDensity::dummyInit( const ProcessorGroup* pc,
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
    CCVariable<double> ParticleDensity;
    
    constCCVariable<double> oldModelTerm;
    constCCVariable<double> oldGasModelTerm;
    constCCVariable<double> oldParticleDensity;

    new_dw->allocateAndPut( ModelTerm,       d_modelLabel,    matlIndex, patch );
    new_dw->allocateAndPut( GasModelTerm,    d_gasLabel,      matlIndex, patch ); 
    new_dw->allocateAndPut( ParticleDensity, d_density_label, matlIndex, patch );

    old_dw->get( oldModelTerm,       d_modelLabel,    matlIndex, patch, gn, 0 );
    old_dw->get( oldGasModelTerm,    d_gasLabel,      matlIndex, patch, gn, 0 );
    old_dw->get( oldParticleDensity, d_density_label, matlIndex, patch, gn, 0 );
    
    ModelTerm.copyData(oldModelTerm);
    GasModelTerm.copyData(oldGasModelTerm);
    ParticleDensity.copyData(oldParticleDensity);

    if( d_constantLength ) {
      CCVariable<double> particle_length;
      new_dw->allocateAndPut( particle_length, d_length_label, matlIndex, patch );
      particle_length.initialize(d_length_constant_value);
    }

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
ParticleDensity::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ParticleDensity::initVars";
  Task* tsk = scinew Task(taskname, this, &ParticleDensity::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes( d_density_label );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
ParticleDensity::initVars( const ProcessorGroup * pc, 
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

    CCVariable<double> density;
    new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
    density.initialize(0.0);

  }
}

