#include <CCA/Components/Arches/PropertyModels/ScalarVarianceScaleSim.h>
#include <CCA/Components/Arches/Filter.h>

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
ScalarVarianceScaleSim::ScalarVarianceScaleSim( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ScalarVarianceScaleSim::~ScalarVarianceScaleSim( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }

  delete _filter; 
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  bool use_old_filter; 

  db->require( "mixture_fraction_label", _mf_label_name ); 
  db->require( "density_label", _density_label_name ); 
  db->getWithDefault( "use_old_filter", use_old_filter, true ); 

  _filter = scinew Filter( use_old_filter );

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "ScalarVarianceScaleSim::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ScalarVarianceScaleSim::computeProp, time_substep ); 

  _density_label = 0; 
  _mf_label = 0; 
  _density_label = VarLabel::find( _density_label_name ); 
  _mf_label      = VarLabel::find( _mf_label_name ); 

  if ( _mf_label == 0 ){ 
    throw InvalidValue("Error: Cannot match mixture fraction name with label.",__FILE__, __LINE__);             
  } 
  if ( _density_label == 0 ){ 
    throw InvalidValue("Error: Cannot match density name with label.",__FILE__, __LINE__);             
  } 

  if ( time_substep == 0 ){ 

    tsk->computes( _prop_label );
    tsk->requires( Task::OldDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::OldDW, _density_label, Ghost::None, 0 ); 

  } else { 

    tsk->modifies( _prop_label ); 
    tsk->requires( Task::NewDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::NewDW, _density_label, Ghost::None, 0 ); 

  } 

  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 

}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::computeProp(const ProcessorGroup* pc, 
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

    Array3<double> filterRho(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhi(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhiSqr(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRho.initialize(0.0);
    filterRhoPhi.initialize(0.0);
    filterRhoPhiSqr.initialize(0.0);

    CCVariable<double> prop; 
    constCCVariable<double> density; 
    constCCVariable<double> mf; 

    if ( time_substep == 0 ) { 
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      prop.initialize(0.0); 

      old_dw->get( mf,      _mf_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      old_dw->get( density, _density_label, matlIndex, patch, Ghost::None, 1 ); 
    } else { 
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 
      new_dw->get( mf,      _mf_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      new_dw->get( density, _density_label, matlIndex, patch, Ghost::None, 1 ); 
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
void ScalarVarianceScaleSim::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::dummyInit( const ProcessorGroup* pc, 
                                            const PatchSubset* patches, 
                                            const MaterialSubset* matls, 
                                            DataWarehouse* old_dw, 
                                            DataWarehouse* new_dw )
{
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ScalarVarianceScaleSim::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ScalarVarianceScaleSim::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::initialize( const ProcessorGroup* pc, 
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
