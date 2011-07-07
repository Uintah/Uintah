#include <CCA/Components/Arches/PropertyModels/ExtentRxn.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
ExtentRxn::ExtentRxn( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // additional local labels as needed by this class (delete this if it isn't used): 
  std::string name = prop_name + "_stripped"; 
  _strip_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels.push_back( _strip_label ); 

  //Model must be evaluated after the table look up: 
  _before_table_lookup = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ExtentRxn::~ExtentRxn( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void ExtentRxn::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  db->require( "fuel_mass_fraction", _fuel_mass_frac ); 
  db->require( "scalar_label", _scalar_name ); 
  db->require( "mix_frac_label", _mixture_fraction_name); 
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ExtentRxn::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "ExtentRxn::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ExtentRxn::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 
      tsk->computes( _strip_label ); 

    } else {

      tsk->modifies( _prop_label ); 
      tsk->modifies( _strip_label );

    }

    const VarLabel* the_label = VarLabel::find(_mixture_fraction_name); 
    tsk->requires( Task::NewDW, the_label, Ghost::None, 0 ); 
    the_label = VarLabel::find(_scalar_name);
    tsk->requires( Task::NewDW, the_label, Ghost::None, 0 ); 

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ExtentRxn::computeProp(const ProcessorGroup* pc, 
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

    CCVariable<double> extent; 
    CCVariable<double> strip; 
    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){
      new_dw->getModifiable( extent, _prop_label, matlIndex, patch ); 
      new_dw->getModifiable( strip, _strip_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( extent, _prop_label, matlIndex, patch ); 
      new_dw->allocateAndPut( strip, _strip_label, matlIndex, patch ); 
      extent.initialize(0.0); 
      strip.initialize(0.0); 
    }

    constCCVariable<double> local_mf; 
    constCCVariable<double> local_scalar; 
    const VarLabel* the_label = VarLabel::find(_mixture_fraction_name); 
    new_dw->get( local_mf, the_label, matlIndex, patch, Ghost::None, 0 ); 
    the_label = VarLabel::find(_scalar_name); 
    new_dw->get( local_scalar, the_label, matlIndex, patch, Ghost::None, 0 ); 

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double hc_wo_rxn = local_mf[c] * _fuel_mass_frac; 
      if ( local_scalar[c] > hc_wo_rxn )
        hc_wo_rxn = local_scalar[c]; 

      double small = 1e-16; 
      if ( hc_wo_rxn > small ) {

        strip[c] = local_scalar[c] / hc_wo_rxn; 
        extent[c] = 1.0 - strip[c]; 

      } else {

        strip[c] = 0.0;
        extent[c] = 1.0; 

      }
    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void ExtentRxn::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "ExtentRxn::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ExtentRxn::dummyInit);
  tsk->computes(_prop_label); 
  tsk->computes(_strip_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void ExtentRxn::dummyInit( const ProcessorGroup* pc, 
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
    CCVariable<double> strip; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    old_dw->get( old_prop, _prop_label, matlIndex, patch, Ghost::None, 0); 
    new_dw->allocateAndPut( strip, _strip_label, matlIndex, patch );
    strip.initialize(0.0); 

    //prop.initialize(0.0); <--- Careful, don't reinitialize if you don't want to 
    prop.copyData( old_prop );

  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void ExtentRxn::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ExtentRxn::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ExtentRxn::initialize);
  tsk->computes(_prop_label); 
  tsk->computes(_strip_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ExtentRxn::initialize( const ProcessorGroup* pc, 
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
    CCVariable<double> strip; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 

    new_dw->allocateAndPut( strip, _strip_label, matlIndex, patch ); 
    strip.initialize(0.0); 

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

  }
}
