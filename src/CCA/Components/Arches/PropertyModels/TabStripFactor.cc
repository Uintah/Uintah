#include <CCA/Components/Arches/PropertyModels/TabStripFactor.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
TabStripFactor::TabStripFactor( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{

  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  _before_table_lookup = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
TabStripFactor::~TabStripFactor( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void TabStripFactor::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  db->require("X", _X); 
  db->require("Y", _Y); 
  _M_CO2 = 44.0; 
  _M_HC  = _X*12.0 + _Y; 
  db->require( "fuel_mass_fraction", _HC_F1 ); 
  db->getWithDefault( "co2_label", _co2_label, string("CO2") ); 
  db->getWithDefault( "ch4_label",_ch4_label, string("CH4") ); 
  db->require( "mix_frac_label", _f_label ); 
  db->getWithDefault( "small", _small, 1e-8 ); 
  
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void TabStripFactor::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "TabStripFactor::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &TabStripFactor::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 

    } else {

      tsk->modifies( _prop_label ); 

    }

    const VarLabel* the_label = VarLabel::find(_co2_label);
    tsk->requires( Task::NewDW, the_label, Ghost::None, 0 ); 
    the_label = VarLabel::find(_f_label); 
    tsk->requires( Task::NewDW, the_label, Ghost::None, 0 ); 

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void TabStripFactor::computeProp(const ProcessorGroup* pc, 
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

    const VarLabel* the_label = VarLabel::find(_co2_label);
    constCCVariable<double> co2; 
    new_dw->get( co2, the_label, matlIndex, patch, Ghost::None, 0 ); 
    the_label = VarLabel::find(_f_label);
    constCCVariable<double> f; 
    new_dw->get( f, the_label, matlIndex, patch, Ghost::None, 0 ); 

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double hc_wo_rxn = f[c] * _HC_F1; 
      double factor = _M_HC/_M_CO2; 

      if ( co2[c] > hc_wo_rxn ) 
        hc_wo_rxn = co2[c]; 

      if ( hc_wo_rxn > _small ){
        prop[c] = 1.0 - factor * co2[c] / hc_wo_rxn; 
      } else 
        prop[c] = 0.0; 

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void TabStripFactor::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "TabStripFactor::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &TabStripFactor::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void TabStripFactor::dummyInit( const ProcessorGroup* pc, 
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
void TabStripFactor::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "TabStripFactor::initialize"; 

  Task* tsk = scinew Task(taskname, this, &TabStripFactor::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void TabStripFactor::initialize( const ProcessorGroup* pc, 
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
