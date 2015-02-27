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

CoalGasMomentum::CoalGasMomentum( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<Vector>::getTypeDescription() ); 

  _source_grid_type = CCVECTOR_SRC; 
}

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

  Task::WhichDW which_dw; 
  if (timeSubStep == 0 ) { 

    tsk->computes(_src_label);
    which_dw = Task::OldDW; 

  } else {

    tsk->modifies(_src_label); 
    which_dw = Task::NewDW; 

  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  CoalModelFactory& modelFactory = CoalModelFactory::self(); 
  
  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){

    std::string model_name = "xdragforce"; 
    std::string node;  
    std::stringstream out; 
    out << iqn; 
    node = out.str(); 
    model_name += "_qn";
    model_name += node; 

    ModelBase& modelx = modelFactory.retrieve_model( model_name ); 

    const VarLabel* tempgasLabel_x = modelx.getGasSourceLabel();
    tsk->requires( which_dw, tempgasLabel_x, Ghost::None, 0 );

    model_name = "ydragforce"; 
    model_name += "_qn";
    model_name += node;

    ModelBase& modely = modelFactory.retrieve_model( model_name );

    const VarLabel* tempgasLabel_y = modely.getGasSourceLabel();
    tsk->requires( which_dw, tempgasLabel_y, Ghost::None, 0 );

    model_name = "zdragforce";
    model_name += "_qn";
    model_name += node;

    ModelBase& modelz = modelFactory.retrieve_model( model_name );

    const VarLabel* tempgasLabel_z = modelz.getGasSourceLabel();
    tsk->requires( which_dw, tempgasLabel_z, Ghost::None, 0 );

  }
  
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

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

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 
    
    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    CoalModelFactory& modelFactory = CoalModelFactory::self(); 
    
    CCVariable<Vector> dragSrc; 
    DataWarehouse* which_dw; 
    if ( timeSubStep == 0 ){ 
      new_dw->allocateAndPut( dragSrc, _src_label, matlIndex, patch );
      dragSrc.initialize(Vector(0.,0.,0.));
      which_dw = old_dw; 
    } else { 
      new_dw->getModifiable( dragSrc, _src_label, matlIndex, patch ); 
      which_dw = new_dw; 
    }
    
    for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){

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
      which_dw->get( qn_gas_xdrag, XDragGasLabel, matlIndex, patch, gn, 0 );

      constCCVariable<double> qn_gas_ydrag;
      model_name = "ydragforce";
      model_name += "_qn";
      model_name += node;

      ModelBase& modely = modelFactory.retrieve_model( model_name );

      const VarLabel* YDragGasLabel = modely.getGasSourceLabel();
      which_dw->get( qn_gas_ydrag, YDragGasLabel, matlIndex, patch, gn, 0 );

      constCCVariable<double> qn_gas_zdrag;
      model_name = "zdragforce";
      model_name += "_qn";
      model_name += node;

      ModelBase& modelz = modelFactory.retrieve_model( model_name );

      const VarLabel* ZDragGasLabel = modelz.getGasSourceLabel();
      which_dw->get( qn_gas_zdrag, ZDragGasLabel, matlIndex, patch, gn, 0 );


      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 
        qn_gas_drag = Vector(qn_gas_xdrag[c],qn_gas_ydrag[c],qn_gas_zdrag[c]);

        dragSrc[c] += qn_gas_drag; // All the work is performed in Drag model

       }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasMomentum::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasMomentum::initialize"; 

  Task* tsk = scinew Task(taskname, this, &CoalGasMomentum::initialize);

  tsk->computes(_src_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
CoalGasMomentum::initialize( const ProcessorGroup* pc, 
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

    CCVariable<Vector> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(Vector(0.0,0.0,0.0)); 

  }
}




