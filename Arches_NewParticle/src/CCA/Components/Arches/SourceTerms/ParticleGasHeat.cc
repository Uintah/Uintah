#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ParticleGasHeat.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

ParticleGasHeat::ParticleGasHeat( std::string src_name, 
                                  vector<std::string> label_names, 
                                  SimulationStateP& shared_state ) 
: SourceTermBase( src_name, shared_state, label_names )
{
  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

ParticleGasHeat::~ParticleGasHeat()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ParticleGasHeat::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require( "heat_model_name", _heat_model_name ); 

  _source_type = CC_SRC; 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
ParticleGasHeat::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ParticleGasHeat::eval";
  Task* tsk = scinew Task(taskname, this, &ParticleGasHeat::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;
  }

  if( timeSubStep == 0 ) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  CoalModelFactory& coal_model_factory = CoalModelFactory::self(); 

  // only require particle temperature labels
  // if there is actually a particle heat transfer model
  if( coal_model_factory.useHeatTransferModel() ) {

    DQMOMEqnFactory& dqmom_factory  = DQMOMEqnFactory::self(); 
    for( int iqn=0; iqn < dqmom_factory.get_quad_nodes(); ++iqn ) {
      HeatTransfer* heat_xfer_model = coal_model_factory.getHeatTransferModel( iqn );
      tsk->requires( Task::NewDW, heat_xfer_model->getGasSourceLabel(), Ghost::None, 0);
    }

  }

  /*
  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
    std::string weight_name = "w_qn";
    std::string model_name = _heat_model_name; 
    std::string node;  
    std::stringstream out; 
    out << iqn; 
    node = out.str(); 
    weight_name += node; 
    model_name += "_qn";
    model_name += node; 

    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( weight_name );

    const VarLabel* tempLabel_w = eqn.getTransportEqnLabel();
    tsk->requires( Task::OldDW, tempLabel_w, Ghost::None, 0 ); 

    ModelBase& model = modelFactory.retrieve_model( model_name ); 
    
    const VarLabel* tempLabel_m = model.getModelLabel(); 
    tsk->requires( Task::OldDW, tempLabel_m, Ghost::None, 0 );

    const VarLabel* tempgasLabel_m = model.getGasSourceLabel();
    tsk->requires( Task::OldDW, tempgasLabel_m, Ghost::None, 0 );

  }
  */

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ParticleGasHeat::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> heatSrc; 
    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( heatSrc, _src_label, matlIndex, patch );
    } else {
      new_dw->getModifiable( heatSrc, _src_label, matlIndex, patch ); 
    }
    heatSrc.initialize(0.0);

    CoalModelFactory& coal_model_factory = CoalModelFactory::self(); 

    if( coal_model_factory.useHeatTransferModel() ) {

      DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self(); 
      int numEnvironments = dqmom_factory.get_quad_nodes();

      // create a vector to hold heat xfer model constCCVariables
      vector< constCCVariable<double>* > heatxferCCVars(numEnvironments);

      // populate this vector with constCCVariables associated with heat xfer model
      for( int iqn=0; iqn < numEnvironments; ++iqn ) {
        heatxferCCVars[iqn] = scinew constCCVariable<double>;
        HeatTransfer* heat_xfer_model = coal_model_factory.getHeatTransferModel(iqn);
        new_dw->get( *(heatxferCCVars[iqn]), heat_xfer_model->getGasSourceLabel(), matlIndex, patch, gn, 0);
      }

      // next, create the net source term by adding source terms for each quad node together
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 

        double running_sum = 0.0;
        for( vector< constCCVariable<double>* >::iterator iHT = heatxferCCVars.begin();
             iHT != heatxferCCVars.end(); ++iHT ) {
          running_sum += (**iHT)[c];
        }

        heatSrc[c] = running_sum;

      }

      // finally, delete constCCVariables created on the stack
      for( vector< constCCVariable<double>* >::iterator ii = heatxferCCVars.begin(); 
           ii != heatxferCCVars.end(); ++ii ) {
        delete *ii;
      }

    }

  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
ParticleGasHeat::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ParticleGasHeat::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ParticleGasHeat::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
ParticleGasHeat::dummyInit( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}




