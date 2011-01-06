#include <CCA/Components/Arches/SourceTerms/DevolMixtureFraction.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

DevolMixtureFraction::DevolMixtureFraction( std::string srcName, 
                                            SimulationStateP& sharedState,
                                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{
  _label_sched_init = false; 
  _src_label = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 

  _source_type = CC_SRC; 
}

DevolMixtureFraction::~DevolMixtureFraction()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DevolMixtureFraction::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require( "devol_model_name", d_devolModelName ); 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
DevolMixtureFraction::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DevolMixtureFraction::eval";
  Task* tsk = scinew Task(taskname, this, &DevolMixtureFraction::computeSource, timeSubStep);

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

  CoalModelFactory& coalFactory = CoalModelFactory::self(); 

  if( coalFactory.useDevolatilizationModel() ) {

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    for( int iqn=0; iqn < dqmomFactory.get_quad_nodes(); ++iqn ) {
      Devolatilization* devol_model = coalFactory.getDevolatilizationModel(iqn);
      tsk->requires( Task::NewDW, devol_model->getGasSourceLabel(), Ghost::None, 0);
    }

  }


  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
DevolMixtureFraction::computeSource( const ProcessorGroup* pc, 
                                     const PatchSubset* patches, 
                                     const MaterialSubset* matls, 
                                     DataWarehouse* old_dw, 
                                     DataWarehouse* new_dw, 
                                     int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> devolSrc; 
    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( devolSrc, _src_label, matlIndex, patch );
    } else {
      new_dw->getModifiable( devolSrc, _src_label, matlIndex, patch ); 
    } 
    devolSrc.initialize(0.0);
    
    CoalModelFactory& coalFactory = CoalModelFactory::self(); 

    if( coalFactory.useDevolatilizationModel() ) {

      DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
      int numEnvironments = dqmomFactory.get_quad_nodes();

      // create a vector to hold devol model constCCVariables
      vector< constCCVariable<double>* > devolCCVars(numEnvironments);

      // populate this vector with constCCVariables associated with heat xfer model
      for( int iqn=0; iqn < numEnvironments; ++iqn ) {
        devolCCVars[iqn] = scinew constCCVariable<double>;
        Devolatilization* devol_model = coalFactory.getDevolatilizationModel(iqn);
        new_dw->get( *(devolCCVars[iqn]), devol_model->getGasSourceLabel(), matlIndex, patch, gn, 0);
      }

      // next, create the net source term by adding source terms for each quad node together
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 

        double running_sum = 0.0;
        for( vector< constCCVariable<double>* >::iterator iD = devolCCVars.begin(); iD != devolCCVars.end(); ++iD ) {
          running_sum += (**iD)[c];
        }

        devolSrc[c] = running_sum;

      }
      
      // finally, delete constCCVariables created on the stack
      for( vector< constCCVariable<double>* >::iterator ii = devolCCVars.begin(); ii != devolCCVars.end(); ++ii ) {
        delete *ii;
      }

    }

  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
DevolMixtureFraction::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DevolMixtureFraction::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &DevolMixtureFraction::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
DevolMixtureFraction::dummyInit( const ProcessorGroup* pc, 
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
    //constCCVariable<double> old_src;
    //old_dw->get(old_src, _src_label, matlIndex, patch, Ghost::None, 0 );
    //src.copyData(old_src);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      tempVar.initialize(0.0);
    }
  }
}

