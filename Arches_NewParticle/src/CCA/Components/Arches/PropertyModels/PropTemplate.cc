#include <CCA/Components/Arches/PropertyModels/CLASSNAME.h>

// Instructions: 
//  1) Make sure you add doxygen comments!!
//  2) If this is not a CCVariable, then either replace with the appropriate 
//     type or use the templated template.  
//  2) Do a find and replace on CLASSNAME to change the your class name 
//  3) Add implementaion details of your property. 
//  4) Here is a brief checklist: 
//     a) Any extra grid variables for this property need to be 
//        given VarLabels in the constructor
//     b) Any extra grid variable VarLabels need to be destroyed
//        in the local destructor
//     c) Add your input file details in problemSetup
//     d) Add actual calculation of property in computeProp. 
//     e) Make sure that you dummyInit any new variables that require OldDW 
//        values.
//     f) Make sure that _before_table_lookup is set propertly for this model.
//        See _before_table_lookup variable. 
//   5) Please clean up unused code from this template in your final version
//   6) Please add comments to this list as you see fit to help the next person

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
CLASSNAME::CLASSNAME( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // additional local labels as needed by this class (delete this if it isn't used): 
  std::string name = "something"; 
  _something_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() ); // Note: you need to add the label to the .h file
  _extra_local_labels.push_back( _something_label ); 

  // and so on ....
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
CLASSNAME::~CLASSNAME( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }
  // Clean up anything else here ... 
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void CLASSNAME::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void CLASSNAME::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "CLASSNAME::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &CLASSNAME::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 

    } else {

      tsk->modifies( _prop_label ); 

    }

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void CLASSNAME::computeProp(const ProcessorGroup* pc, 
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

    CCVariable<double> prop; 
    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      prop.initialize(0.0); 
    }

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      prop[*iter] = 0.0; // <--- do something here. 

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void CLASSNAME::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "CLASSNAME::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &CLASSNAME::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void CLASSNAME::dummyInit( const ProcessorGroup* pc, 
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
void CLASSNAME::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CLASSNAME::initialize"; 

  Task* tsk = scinew Task(taskname, this, &CLASSNAME::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void CLASSNAME::initialize( const ProcessorGroup* pc, 
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
