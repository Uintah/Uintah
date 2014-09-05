#include <CCA/Components/Arches/PropertyModels/ScalarDissipation.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
ScalarDissipation::ScalarDissipation( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() );  
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 
  
}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ScalarDissipation::~ScalarDissipation( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    
    VarLabel::destroy( *iter ); 
    
  }
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void ScalarDissipation::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 
  
  db->require( "grad_mixfrac2_label", _gradmf2_name );
  db->require( "D", _D ); 
  
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ScalarDissipation::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  
  _gradmf2_label = 0;
  _gradmf2_label = VarLabel::find( _gradmf2_name ); 
  
  if ( _gradmf2_label == 0 ){ 
    throw InvalidValue("Error: Cannot match grad mixture fraction squared name with label.",__FILE__, __LINE__);             
  } 
  
  std::string taskname = "ScalarDissipation::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ScalarDissipation::computeProp, time_substep ); 
  
  tsk->modifies( _prop_label );
  if ( time_substep == 0 ){ 
    
    tsk->requires( Task::OldDW, _gradmf2_label, Ghost::None, 0 ); 
  } else { 
    
    tsk->requires( Task::NewDW, _gradmf2_label, Ghost::None, 0 ); 
  } 
  
  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ScalarDissipation::computeProp(const ProcessorGroup* pc, 
                                      const PatchSubset* patches, 
                                      const MaterialSubset* matls, 
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw, 
                                      int time_substep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 
    
    CCVariable<double>      prop; 
    constCCVariable<double> gradmf2;
    
    if ( time_substep == 0 ) { 
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      old_dw->get( gradmf2, _gradmf2_label, matlIndex, patch, Ghost::None, 0 ); 
      prop.initialize(0.0); 
    } else { 
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch, Ghost::None, 0 ); 
      new_dw->get( gradmf2, _gradmf2_label, matlIndex, patch, Ghost::None, 0); 
    } 
    
    CellIterator iter = patch->getCellIterator(); 
        
    for (iter.begin(); !iter.done(); iter++){
      IntVector c = *iter;       
      prop[c] = 2.0 * ( _D  ) * gradmf2[c];
    }
    
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void ScalarDissipation::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ScalarDissipation::initialize"; 
  
  Task* tsk = scinew Task(taskname, this, &ScalarDissipation::initialize);
  tsk->computes(_prop_label); 
  
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ScalarDissipation::initialize( const ProcessorGroup* pc, 
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
