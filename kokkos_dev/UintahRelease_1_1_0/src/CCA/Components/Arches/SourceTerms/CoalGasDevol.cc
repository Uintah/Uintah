#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>

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
{}

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

  timeSubStep; 

  if (timeSubStep == 0 && !d_labelSchedInit) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_srcLabel);
  } else {
    tsk->modifies(d_srcLabel); 
  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  ModelFactory& modelFactory = ModelFactory::self(); 
  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
    std::string wght_name = "w_qn";
    std::string model_name = d_devolModelName; 
    std::string node;  
    std::stringstream out; 
    out << iqn; 
    node = out.str(); 
    wght_name += node; 
    model_name += "_qn";
    model_name += node; 

    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( wght_name );

    const VarLabel* tempLabel_w = eqn.getTransportEqnLabel();
    tsk->requires( Task::OldDW, tempLabel_w, Ghost::None, 0 ); 

    ModelBase& temp_model = modelFactory.retrieve_model( model_name ); 
    KobayashiSarofimDevol& model = dynamic_cast<KobayashiSarofimDevol&>(temp_model);
    
    const VarLabel* tempLabel_m = model.getModelLabel(); 
    tsk->requires( Task::OldDW, tempLabel_m, Ghost::None, 0 );

    const VarLabel* tempgasLabel_m = model.getGasRateLabel();
    tsk->requires( Task::OldDW, tempgasLabel_m, Ghost::None, 0 );

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

    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    ModelFactory& modelFactory = ModelFactory::self(); 

    CCVariable<double> devolSrc; 
    if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
      new_dw->getModifiable( devolSrc, d_srcLabel, matlIndex, patch ); 
      devolSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( devolSrc, d_srcLabel, matlIndex, patch );
      devolSrc.initialize(0.0);
    } 


    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter;


      for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
        std::string model_name = d_devolModelName; 
        std::string node;  
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        model_name += "_qn";
        model_name += node;

        ModelBase& temp_model = modelFactory.retrieve_model( model_name ); 
        KobayashiSarofimDevol& model = dynamic_cast<KobayashiSarofimDevol&>(temp_model); 

        constCCVariable<double> qn_gas_devol;
        const VarLabel* gasModelLabel = model.getGasRateLabel(); 
        old_dw->get( qn_gas_devol, gasModelLabel, matlIndex, patch, gn, 0 );

        devolSrc[c] += qn_gas_devol[c]; // All the work is performed in Kobayashi/Sarofim model
      }
    }
  }
}


