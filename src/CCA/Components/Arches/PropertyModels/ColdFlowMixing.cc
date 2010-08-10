#include <CCA/Components/Arches/PropertyModels/ColdFlowMixing.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
ColdFlowMixing::ColdFlowMixing( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label        = VarLabel::create( "density", CCVariable<double>::getTypeDescription() ); 

  // additional local labels as needed by this class (delete this if it isn't used): 
  std::string name = "temperature"; 
  _temperature_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() ); // Note: you need to add the label to the .h file
  _extra_local_labels.push_back( _temperature_label ); 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ColdFlowMixing::~ColdFlowMixing( )
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
void ColdFlowMixing::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  // stream 1 "fuel"
  db->findBlock( "fuel" )->require( "T", _T_f); 
  db->findBlock( "fuel" )->require( "rho", _rho_f); 
  // stream 2 "oxidizer"
  db->findBlock( "oxidizer" )->require( "T", _T_o); 
  db->findBlock( "oxidizer" )->require( "rho", _rho_o); 

  // name of the mixture fraction label
  db->require( "mix_frac_label", _f_name ); 

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ColdFlowMixing::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "ColdFlowMixing::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ColdFlowMixing::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 

      for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        tsk->computes( *iter ); 
      }

    } else {

      tsk->modifies( _prop_label ); 
      for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        tsk->modifies( *iter ); 
      }

    }

    const VarLabel* f_label = VarLabel::find( _f_name ); 
    tsk->requires( Task::NewDW, f_label, Ghost::None, 0 ); 
    const VarLabel* dencp_label = VarLabel::find( "densityCP" ); 
    tsk->modifies( dencp_label ); 

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ColdFlowMixing::computeProp(const ProcessorGroup* pc, 
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

    CCVariable<double> density; 
    CCVariable<double> temperature; 
    constCCVariable<double> f; 
    CCVariable<double> denCP; 
    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){
      new_dw->getModifiable( density, _prop_label, matlIndex, patch ); 
      new_dw->getModifiable( temperature, _temperature_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( density, _prop_label, matlIndex, patch ); 
      new_dw->allocateAndPut( temperature, _temperature_label, matlIndex, patch ); 
      density.initialize(0.0); 
      temperature.initialize(0.0); 
    }
    const VarLabel* f_label = VarLabel::find( _f_name ); 
    new_dw->get( f, f_label, matlIndex, patch, Ghost::None, 0 ); 
    const VarLabel* dencp_label = VarLabel::find( "densityCP" ); 
    new_dw->getModifiable( denCP, dencp_label, matlIndex, patch ); 

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double inv_den = ( f[c] ) / (_rho_f) + ( 1.0 - f[c] ) / (_rho_o); 
      double inv_T   = ( f[c] ) / (_T_f)   + ( 1.0 - f[c] ) / (_T_o); 

      density[c] = 1.0 / inv_den; 
      temperature[c] = 1.0 / inv_T; 

      // hack for now until Properties.cc disappears
      denCP[c] = density[c]; 

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void ColdFlowMixing::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "ColdFlowMixing::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ColdFlowMixing::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void ColdFlowMixing::dummyInit( const ProcessorGroup* pc, 
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
void ColdFlowMixing::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ColdFlowMixing::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ColdFlowMixing::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ColdFlowMixing::initialize( const ProcessorGroup* pc, 
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
