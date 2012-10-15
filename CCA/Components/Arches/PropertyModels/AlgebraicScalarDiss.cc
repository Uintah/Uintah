#include <CCA/Components/Arches/PropertyModels/AlgebraicScalarDiss.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
AlgebraicScalarDiss::AlgebraicScalarDiss( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // additional local labels as needed by this class (delete this if it isn't used): 
  //std::string name = "something"; 
  //_something_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() ); // Note: you need to add the label to the .h file
  //_extra_local_labels.push_back( _something_label ); 
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
AlgebraicScalarDiss::~AlgebraicScalarDiss( )
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
void AlgebraicScalarDiss::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  db->require( "mixture_fraction_label", _mf_name );
  db->require( "turbulent_Sc", _Sc_t ); 
  db->require( "D", _D ); 

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{

  _mf_label = 0;
  _mf_label = VarLabel::find( _mf_name ); 
  _mu_t_label = 0;
  _mu_t_label = VarLabel::find( "turb_viscosity" ); 


  if ( _mf_label == 0 ){ 
    throw InvalidValue("Error: Cannot match mixture fraction name with label.",__FILE__, __LINE__);             
  } 
  if ( _mu_t_label == 0 ){ 
    throw InvalidValue("Error: Cannot match turbulent viscosity name with label.",__FILE__, __LINE__);             
  } 

  std::string taskname = "AlgebraicScalarDiss::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &AlgebraicScalarDiss::computeProp, time_substep ); 

  if ( time_substep == 0 ){ 

    tsk->computes( _prop_label );
    tsk->requires( Task::OldDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::OldDW, _mf_label, Ghost::None, 0 ); 

  } else { 

    tsk->modifies( _prop_label ); 
    tsk->requires( Task::NewDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::NewDW, _mu_t_label, Ghost::None, 0 ); 

  } 

  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::computeProp(const ProcessorGroup* pc, 
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
    constCCVariable<double> mf;
    constCCVariable<double> mu_t; 

    if ( time_substep == 0 ) { 
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      old_dw->get( mf, _mf_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      old_dw->get( mu_t, _mu_t_label, matlIndex, patch, Ghost::None, 0 ); 

      prop.initialize(0.0); 
    } else { 
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch, Ghost::None, 0 ); 
      new_dw->get( mf, _mf_label, matlIndex, patch, Ghost::AroundCells, 1); 
      new_dw->get( mu_t, _mu_t_label, matlIndex, patch, Ghost::None, 0 ); 
    } 

    CellIterator iter = patch->getCellIterator(); 

    Vector Dx = patch->dCell(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 
      IntVector cxm = c - IntVector(1,0,0); 
      IntVector cxp = c + IntVector(1,0,0); 
      IntVector cym = c - IntVector(0,1,0); 
      IntVector cyp = c + IntVector(0,1,0); 
      IntVector czm = c - IntVector(0,0,1); 
      IntVector czp = c + IntVector(0,0,1); 

      double gradZ = 1.0/Dx.x() * ( 0.5*(mf[cxm] + mf[cxp]) - mf[c] ) + 
                     1.0/Dx.y() * ( 0.5*(mf[cym] + mf[cyp]) - mf[c] ) + 
                     1.0/Dx.z() * ( 0.5*(mf[czm] + mf[czp]) - mf[c] ); 

      gradZ = std::abs(gradZ);

      double Dt = mu_t[c] / _Sc_t; 
      prop[c] = 2.0 * ( _D + Dt ) * pow(gradZ,2.0);

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::dummyInit( const ProcessorGroup* pc, 
                                            const PatchSubset* patches, 
                                            const MaterialSubset* matls, 
                                            DataWarehouse* old_dw, 
                                            DataWarehouse* new_dw )
{
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "AlgebraicScalarDiss::initialize"; 

  Task* tsk = scinew Task(taskname, this, &AlgebraicScalarDiss::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void AlgebraicScalarDiss::initialize( const ProcessorGroup* pc, 
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
