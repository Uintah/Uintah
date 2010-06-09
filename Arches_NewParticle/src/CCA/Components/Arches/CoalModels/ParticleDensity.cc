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

//  d_densityLabels.resize(numQuadNodes);
  
  std::string density_name = "rhop_qn";

  std::string qnode;
  std::stringstream out;
  out << qn;
  qnode = out.str();

  d_density_label = VarLabel::create( density_name+qnode, CCVariable<double>::getTypeDescription() );
}

ParticleDensity::~ParticleDensity()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ParticleDensity::problemSetup(const ProblemSpecP& params )
{
  // This method is called by problemSetup() in child classes

  ProblemSpecP db = params; 

  const ProblemSpecP params_root = db->getRootNode();
  if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties") ) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("initial_ash_mass", ash_mass);
  } else {
    throw InvalidValue("Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);
  }

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
  d_w_scaling_factor = weight_eqn.getScalingConstant();

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
ParticleDensity::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ParticleDensity::initVars";
  Task* tsk = scinew Task(taskname, this, &ParticleDensity::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );

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
  }
}

/*
void 
ParticleDensity::sched_computeParticleDensity( const LevelP&  level,
                                               SchedulerP&    sched,
                                               int            timeSubStep )
{
  std::string taskname = "ParticleDensity::computeParticleDensity";
  Task* tsk = scinew Task(taskname, this, &ParticleDensity::computeParticleDensity);

  for( vector<VarLabel*>::iterator iLabel = d_densityLabels.begin();
       iLabel != d_densityLabels.end(); ++iLabel ) {
    if( timeSubStep == 0 && !d_labelSchedInit ) {
      tsk->computes(*iLabel);
    } else {
      tsk->modifies(*iLabel);
    }
  }

  if( timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}
*/

/*
void 
ParticleDensity::computeParticleDensity( const ProcessorGroup* pc,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw )
{
  // This method left intentionally blank
}
*/

