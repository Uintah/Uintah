#include <CCA/Components/Arches/PropertyModels/fvSootFromYsoot.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace Uintah;
using namespace std;

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
fvSootFromYsoot::fvSootFromYsoot( std::string prop_name, SimulationStateP& shared_state ) :
  PropertyModelBase( prop_name, shared_state )
{

  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() );

  _before_table_lookup = false;

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
fvSootFromYsoot::~fvSootFromYsoot( )
{
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void fvSootFromYsoot::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb;

  db->require( "opl", _opl );

  db->getWithDefault( "density_label" ,          _den_label_name,    "density"	  );
  db->getWithDefault( "Ysoot_label"   ,          _Ys_label_name ,    "Ysoot"	  );
  db->getWithDefault( "absorption_label",        _absorp_label_name, "absorpIN"	  );
  db->getWithDefault( "temperature_label",	 _T_label_name,	     "temperature");

  db->getWithDefault( "soot_density", _rho_soot, 1950.0);
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void fvSootFromYsoot::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "fvSootFromYsoot::computeProp";
  Task* tsk = scinew Task( taskname, this, &fvSootFromYsoot::computeProp, time_substep );
  Ghost::GhostType  gn  = Ghost::None;

  _den_label    = VarLabel::find( _den_label_name );
  _T_label	= VarLabel::find( _T_label_name);
  _Ys_label     = VarLabel::find( _Ys_label_name );
  _absorp_label = VarLabel::find( _absorp_label_name );

  if ( _den_label == 0 ){
    throw InvalidValue("Error: Cannot find den label in the fv soot function with name: "+_den_label_name,__FILE__,__LINE__);
  } else if ( _Ys_label == 0 ){
    throw InvalidValue("Error: Cannot find Ys label in the fv soot function with name: "+_Ys_label_name,__FILE__,__LINE__);
  } else if ( _absorp_label == 0 ){
    throw InvalidValue("Error: Cannot fine absorp label in the fv soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _T_label ==0 ){
    throw InvalidValue("Error: Cannot fine temperature label in the fv soot function with name: "+_T_label_name,__FILE__,__LINE__);
  }

  if ( time_substep == 0 ) {

    tsk->requires( Task::OldDW, _T_label,    gn, 0 );
    tsk->requires( Task::OldDW, _den_label,  gn, 0 );
    tsk->requires( Task::OldDW, _Ys_label,   gn, 0 );
    
  } else {

    tsk->requires( Task::NewDW, _T_label,    gn, 0 );
    tsk->requires( Task::NewDW, _den_label,  gn, 0 );
    tsk->requires( Task::NewDW, _Ys_label,   gn, 0 );

    tsk->modifies( _absorp_label );

  }

  tsk->modifies( _prop_label );
  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() );

}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void fvSootFromYsoot::computeProp(const ProcessorGroup* pc,
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

    CCVariable<double> soot_vf;
    CCVariable<double> absorp_coef;
    constCCVariable<double> temperature;
    constCCVariable<double> density;
    constCCVariable<double> Ysoot;
    Ghost::GhostType  gn  = Ghost::None;

    if ( time_substep != 0 ){

      new_dw->getModifiable( absorp_coef,    _absorp_label,   matlIndex, patch );
      new_dw->getModifiable( soot_vf,     _prop_label,   matlIndex, patch );

      new_dw->get( temperature,  _T_label,  matlIndex, patch, gn, 0 );
      new_dw->get( density,     _den_label, matlIndex, patch, gn, 0 );
      new_dw->get( Ysoot,       _Ys_label,  matlIndex, patch, gn, 0 );

    } else {
      
      new_dw->allocateAndPut( soot_vf,      _prop_label,     matlIndex, patch );
      new_dw->getModifiable(  absorp_coef,  _absorp_label,   matlIndex, patch );

      soot_vf.initialize(0.0);
      absorp_coef.initialize(0.0);

      old_dw->get( temperature, _T_label,   matlIndex, patch, gn, 0 );
      old_dw->get( density,     _den_label, matlIndex, patch, gn, 0 );
      old_dw->get( Ysoot,       _den_label, matlIndex, patch, gn, 0 );

    }

    CellIterator iter = patch->getCellIterator();

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter;
      soot_vf[c] = density[c] * Ysoot[c] / _rho_soot;

      absorp_coef[c] += 0.01 + std::min( 0.5, (4.0/_opl)*log( 1.0 +
		       350.0 * soot_vf[c] * temperature[c] * _opl));
    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void fvSootFromYsoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "fvSootFromYsoot::initialize";

  _den_label    = VarLabel::find( _den_label_name );
  _Ys_label     = VarLabel::find( _Ys_label_name );
  _absorp_label = VarLabel::find( _absorp_label_name );

  if ( _den_label == 0 ){
    throw InvalidValue("Error: Cannot find den label in the fv soot function with name: "+_den_label_name,__FILE__,__LINE__);
  } else if ( _Ys_label == 0 ){
    throw InvalidValue("Error: Cannot find Ys label in the fv soot function with name: "+_Ys_label_name,__FILE__,__LINE__);
  }

  Task* tsk = scinew Task(taskname, this, &fvSootFromYsoot::initialize);
  tsk->computes(_prop_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void fvSootFromYsoot::initialize( const ProcessorGroup* pc,
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
