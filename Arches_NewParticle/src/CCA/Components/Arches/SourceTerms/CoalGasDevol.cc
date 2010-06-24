#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
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

//---------------------------------------------------------------------------
// Builder:
CoalGasDevolBuilder::CoalGasDevolBuilder(std::string srcName, 
                                         vector<std::string> reqLabelNames, 
                                         SimulationStateP& sharedState)
: SourceTermBuilder(srcName, reqLabelNames, sharedState)
{}

CoalGasDevolBuilder::~CoalGasDevolBuilder(){}

SourceTermBase*
CoalGasDevolBuilder::build(){
  return scinew CoalGasDevol( d_srcName, d_sharedState, d_requiredLabels );
}
// End Builder
//---------------------------------------------------------------------------

CoalGasDevol::CoalGasDevol( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{
  d_srcLabel = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 
}

CoalGasDevol::~CoalGasDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CoalGasDevol::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require( "devol_model_name", d_devolModelName ); 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
CoalGasDevol::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasDevol::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasDevol::computeSource, timeSubStep);

  if (timeSubStep == 0 && !d_labelSchedInit) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_srcLabel);
  } else {
    tsk->modifies(d_srcLabel); 
  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  CoalModelFactory& modelFactory = CoalModelFactory::self(); 

  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
    std::string weight_name = "w_qn";
    std::string model_name = d_devolModelName; 
    std::string node;  
    std::stringstream out; 
    out << iqn; 
    node = out.str(); 
    weight_name += node; 
    model_name += "_qn";
    model_name += node; 

    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( weight_name );

    const VarLabel* tempLabel_w = eqn.getTransportEqnLabel();
    tsk->requires( Task::NewDW, tempLabel_w, Ghost::None, 0 ); 

    ModelBase& model = modelFactory.retrieve_model( model_name ); 
    
    const VarLabel* tempLabel_m = model.getModelLabel(); 
    tsk->requires( Task::NewDW, tempLabel_m, Ghost::None, 0 );

    const VarLabel* tempgasLabel_m = model.getGasSourceLabel();
    tsk->requires( Task::NewDW, tempgasLabel_m, Ghost::None, 0 );

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CoalGasDevol::computeSource( const ProcessorGroup* pc, 
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
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> devolSrc; 
    if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
      new_dw->getModifiable( devolSrc, d_srcLabel, matlIndex, patch ); 
      devolSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( devolSrc, d_srcLabel, matlIndex, patch );
      devolSrc.initialize(0.0);
    } 
    
    CoalModelFactory& modelFactory = CoalModelFactory::self(); 
    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    int numEnvironments = dqmomFactory.get_quad_nodes();

    // vector holding model constCCvariables 
    vector< constCCVariable<double>* > modelCCVars(numEnvironments);
    
    // populate vector holding model constCCVariables
    for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++) {
      modelCCVars[iqn] = scinew constCCVariable<double>;

      std::string model_name = d_devolModelName; 
      std::string node;  
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      model_name += "_qn";
      model_name += node;

      ModelBase& model = modelFactory.retrieve_model( model_name ); 

      new_dw->get( *(modelCCVars[iqn]), model.getGasSourceLabel(), matlIndex, patch, gn, 0 );
    }
        
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double running_sum = 0.0;
      for( vector< constCCVariable<double>* >::iterator iModel = modelCCVars.begin(); 
           iModel != modelCCVars.end(); ++iModel ) {
        running_sum += (**iModel)[c];
      }
      
      devolSrc[c] = running_sum;
    }

    for( vector< constCCVariable<double>* >::iterator i = modelCCVars.begin();
         i != modelCCVars.end(); ++i ) {
      delete *i;
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
CoalGasDevol::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasDevol::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &CoalGasDevol::dummyInit);

  tsk->computes(d_srcLabel);

  for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
CoalGasDevol::dummyInit( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;

    new_dw->allocateAndPut( src, d_srcLabel, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}




