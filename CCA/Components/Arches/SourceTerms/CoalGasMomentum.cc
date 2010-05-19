#include <CCA/Components/Arches/SourceTerms/CoalGasMomentum.h>

#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>

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
CoalGasMomentumBuilder::CoalGasMomentumBuilder(std::string srcName, 
                                         vector<std::string> reqLabelNames, 
                                         SimulationStateP& sharedState)
: SourceTermBuilder(srcName, reqLabelNames, sharedState)
{}

CoalGasMomentumBuilder::~CoalGasMomentumBuilder(){}

SourceTermBase*
CoalGasMomentumBuilder::build(){
  return scinew CoalGasMomentum( d_srcName, d_sharedState, d_requiredLabels );
}
// End Builder
//---------------------------------------------------------------------------

CoalGasMomentum::CoalGasMomentum( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{}

CoalGasMomentum::~CoalGasMomentum()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CoalGasMomentum::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  //db->getWithDefault("constant",d_constant, 0.1); 
  //db->getWithDefault( "drag_model_name", d_dragModelName, "dragforce" );

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
CoalGasMomentum::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{ 
  std::string taskname = "CoalGasMomentum::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasMomentum::computeSource, timeSubStep);

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
    std::string model_name = "xdragforce"; 
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

    ModelBase& modelx = modelFactory.retrieve_model( model_name ); 

    const VarLabel* tempLabel_x = modelx.getModelLabel(); 
    tsk->requires( Task::OldDW, tempLabel_x, Ghost::None, 0 );

    const VarLabel* tempgasLabel_x = modelx.getGasSourceLabel();
    tsk->requires( Task::OldDW, tempgasLabel_x, Ghost::None, 0 );

    model_name = "ydragforce"; 
    model_name += "_qn";
    model_name += node;

    ModelBase& modely = modelFactory.retrieve_model( model_name );

    const VarLabel* tempLabel_y = modely.getModelLabel();
    tsk->requires( Task::OldDW, tempLabel_y, Ghost::None, 0 );

    const VarLabel* tempgasLabel_y = modely.getGasSourceLabel();
    tsk->requires( Task::OldDW, tempgasLabel_y, Ghost::None, 0 );

    model_name = "zdragforce";
    model_name += "_qn";
    model_name += node;

    ModelBase& modelz = modelFactory.retrieve_model( model_name );

    const VarLabel* tempLabel_z = modelz.getModelLabel();
    tsk->requires( Task::OldDW, tempLabel_z, Ghost::None, 0 );

    const VarLabel* tempgasLabel_z = modelz.getGasSourceLabel();
    tsk->requires( Task::OldDW, tempgasLabel_z, Ghost::None, 0 );


  }
  
  for (vector<std::string>::iterator iter = d_requiredLabels.begin(); 
       iter != d_requiredLabels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CoalGasMomentum::computeSource( const ProcessorGroup* pc, 
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
    
    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    CoalModelFactory& modelFactory = CoalModelFactory::self(); 
    
    CCVariable<Vector> dragSrc; 
    if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
      new_dw->getModifiable( dragSrc, d_srcLabel, matlIndex, patch ); 
      dragSrc.initialize(Vector(0.,0.,0.)); 
    } else {
      new_dw->allocateAndPut( dragSrc, d_srcLabel, matlIndex, patch );
      dragSrc.initialize(Vector(0.,0.,0.));
    } 
    
    for (vector<std::string>::iterator iter = d_requiredLabels.begin(); 
         iter != d_requiredLabels.end(); iter++) { 
      //CCVariable<double> temp; 
      //old_dw->get( *iter.... ); 
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

       for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
        //std::string model_name = d_dragModelName; 
        //std::string node;  
        //std::stringstream out; 
        //out << iqn; 
        //node = out.str(); 
        //model_name += "_qn";
        //model_name += node;

        //ModelBase& model = modelFactory.retrieve_model( model_name ); 

        Vector qn_gas_drag;

        constCCVariable<double> qn_gas_xdrag;
        std::string model_name = "xdragforce";
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();
        model_name += "_qn";
        model_name += node;

        ModelBase& modelx = modelFactory.retrieve_model( model_name );

        const VarLabel* XDragGasLabel = modelx.getGasSourceLabel();  
        old_dw->get( qn_gas_xdrag, XDragGasLabel, matlIndex, patch, gn, 0 );

        constCCVariable<double> qn_gas_ydrag;
        model_name = "ydragforce";
        model_name += "_qn";
        model_name += node;

        ModelBase& modely = modelFactory.retrieve_model( model_name );

        const VarLabel* YDragGasLabel = modely.getGasSourceLabel();
        old_dw->get( qn_gas_ydrag, YDragGasLabel, matlIndex, patch, gn, 0 );

        constCCVariable<double> qn_gas_zdrag;
        model_name = "zdragforce";
        model_name += "_qn";
        model_name += node;

        ModelBase& modelz = modelFactory.retrieve_model( model_name );

        const VarLabel* ZDragGasLabel = modelz.getGasSourceLabel();
        old_dw->get( qn_gas_zdrag, ZDragGasLabel, matlIndex, patch, gn, 0 );

        qn_gas_drag = Vector(qn_gas_xdrag[c],qn_gas_ydrag[c],qn_gas_zdrag[c]);

        dragSrc[c] += qn_gas_drag; // All the work is performed in Drag model

       }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
CoalGasMomentum::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasMomentum::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &CoalGasMomentum::dummyInit);

  tsk->computes(d_srcLabel);

  for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
CoalGasMomentum::dummyInit( const ProcessorGroup* pc, 
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




