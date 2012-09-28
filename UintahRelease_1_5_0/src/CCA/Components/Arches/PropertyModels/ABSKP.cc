#include <CCA/Components/Arches/PropertyModels/ABSKP.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
ABSKP::ABSKP( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  // need a predictable name here: 
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = true; // Because it is used in the table lookup

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ABSKP::~ABSKP( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }

//  for( HeatTransferModelMap::iterator i=heatmodels_.begin(); i!=heatmodels_.end(); ++i ){
//      delete i->second;
//  }

  // Clean up anything else here ... 
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void ABSKP::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ABSKP::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "ABSKP::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ABSKP::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 

    } else {

      tsk->modifies( _prop_label ); 

    }

    CoalModelFactory& modelFactory = CoalModelFactory::self();
    heatmodels_ = modelFactory.retrieve_heattransfer_models();
    for( HeatTransferModelMap::iterator iModel = heatmodels_.begin(); iModel != heatmodels_.end(); ++iModel ) {
      const VarLabel* tempabskpLabel = iModel->second->getabskpLabel();
      tsk->requires( Task::OldDW, tempabskpLabel, Ghost::None, 0 );
    }

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ABSKP::computeProp(const ProcessorGroup* pc, 
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw, 
                                    int time_substep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> prop; 
    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    }
    prop.initialize(0.0);

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      for( HeatTransferModelMap::iterator iModel = heatmodels_.begin(); iModel != heatmodels_.end(); ++iModel ) {
        //int modelNode = iModel->second->getquadNode();
        const VarLabel* tempabskpLabel = iModel->second->getabskpLabel();
        constCCVariable<double> qn_abskp;
        old_dw->get( qn_abskp, tempabskpLabel, matlIndex, patch, gn, 0 );
        prop[*iter] += qn_abskp[*iter];
        //cout << "ABSKP: qn " << modelNode << " value " << qn_abskp[*iter] << endl;
      }

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void ABSKP::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "ABSKP::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ABSKP::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void ABSKP::dummyInit( const ProcessorGroup* pc, 
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

    CCVariable<double> prop; 
    constCCVariable<double> old_prop; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    old_dw->get( old_prop, _prop_label, matlIndex, patch, Ghost::None, 0); 

    //prop.initialize(0.0); <--- Careful, don't reinitialize if you don't want to 
    prop.copyData( old_prop );

  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void ABSKP::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ABSKP::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ABSKP::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ABSKP::initialize( const ProcessorGroup* pc, 
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

    CCVariable<double> prop; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

  }
}
