#include <CCA/Components/Arches/SourceTerms/CharOxidationMixtureFraction.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
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
CharOxidationMixtureFractionBuilder::CharOxidationMixtureFractionBuilder(std::string srcName, 
                                                                         vector<std::string> reqLabelNames, 
                                                                         SimulationStateP& sharedState)
: SourceTermBuilder(srcName, reqLabelNames, sharedState)
{}

CharOxidationMixtureFractionBuilder::~CharOxidationMixtureFractionBuilder(){}

SourceTermBase*
CharOxidationMixtureFractionBuilder::build(){
  return scinew CharOxidationMixtureFraction( d_srcName, d_sharedState, d_requiredLabels );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationMixtureFraction::CharOxidationMixtureFraction( std::string srcName, 
                                                            SimulationStateP& sharedState,
                                                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{
  d_srcLabel = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 
}

CharOxidationMixtureFraction::~CharOxidationMixtureFraction()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CharOxidationMixtureFraction::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 

  db->require( "char_model_name", d_charModelName ); 

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  CoalModelFactory& coalFactory = CoalModelFactory::self();

  for( int iNode = 0; iNode < dqmomFactory.get_quad_nodes(); ++iNode ) {
    std::stringstream out;
    out << iNode;
    string node = out.str();
    string tempName = d_charModelName + "_qn" + node;

    CharOxidation* char_model = dynamic_cast<CharOxidation*>(&coalFactory.retrieve_model(tempName)); 

    for( ProblemSpecP db_oxidizer = db->findBlock("oxidizer"); db_oxidizer != 0; db_oxidizer = db_oxidizer->findNextBlock("oxidizer") ) {
      string species_name;
      db_oxidizer->get("species",species_name);
      GasModelLabels_.push_back( char_model->getGasModelLabel(species_name) );
    }
  }

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
CharOxidationMixtureFraction::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CharOxidationMixtureFraction::computeSource";
  Task* tsk = scinew Task(taskname, this, &CharOxidationMixtureFraction::computeSource, timeSubStep);

  if (timeSubStep == 0 && !d_labelSchedInit) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {
    tsk->computes(d_srcLabel);
  } else {
    tsk->modifies(d_srcLabel); 
  }

  for( vector<const VarLabel*>::iterator iGasModel = GasModelLabels_.begin(); iGasModel != GasModelLabels_.end(); ++iGasModel ) {
    tsk->requires( Task::NewDW, *iGasModel, Ghost::None, 0 ); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CharOxidationMixtureFraction::computeSource( const ProcessorGroup* pc, 
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

    CCVariable<double> mixFracSrc; 
    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( mixFracSrc, d_srcLabel, matlIndex, patch );
      mixFracSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( mixFracSrc, d_srcLabel, matlIndex, patch ); 
      mixFracSrc.initialize(0.0);
    } 

    vector< constCCVariable<double>* > gasModelCCVars;

    int zz=0;
    for( vector<const VarLabel*>::iterator iLabel = GasModelLabels_.begin(); iLabel != GasModelLabels_.end(); ++iLabel, ++zz ) {
      gasModelCCVars.push_back( scinew constCCVariable<double> );
      new_dw->get( *(gasModelCCVars[zz]), *iLabel, matlIndex, patch, gn, 0 );
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double running_sum = 0.0;
      for( vector< constCCVariable<double>* >::iterator iGasModel = gasModelCCVars.begin(); 
           iGasModel != gasModelCCVars.end(); ++iGasModel ) {
        running_sum += (**iGasModel)[c];
      }

      mixFracSrc[c] = running_sum;
    }

    // now delete CCVariables created on the heap with the "scinew" operator
    for( vector< constCCVariable<double>* >::iterator i = gasModelCCVars.begin(); i != gasModelCCVars.end(); ++i ) {
      delete *i;
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
CharOxidationMixtureFraction::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CharOxidationMixtureFraction::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &CharOxidationMixtureFraction::dummyInit);

  tsk->computes(d_srcLabel);

  for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
CharOxidationMixtureFraction::dummyInit( const ProcessorGroup* pc, 
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




