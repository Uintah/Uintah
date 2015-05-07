#include <CCA/Components/Arches/PropertyModels/EmpSoot.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace Uintah; 
using namespace std;

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
EmpSoot::EmpSoot( std::string prop_name, SimulationStateP& shared_state ) : 
  PropertyModelBase( prop_name, shared_state ), 
  _cmw(12.0)
{

  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 
  
  _before_table_lookup = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
EmpSoot::~EmpSoot( )
{
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void EmpSoot::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 
  
  db->require( "carbon_content_fuel", _carb_content_fuel );
  db->require( "carbon_content_ox",   _carb_content_ox );
  db->require( "opl", _opl );  

  db->getWithDefault( "density_label",          _den_label_name, "density"); 
  db->getWithDefault( "absorption_label",       _absorp_label_name, "absorpIN"); 
  db->getWithDefault( "temperature_label",      _T_label_name, "temperature"); 
  db->getWithDefault( "mixture_fraction_label", _f_label_name, "mixture_fraction"); 

  db->getWithDefault( "soot_density", _rho_soot, 1950.0); 
  db->getWithDefault( "E_cr", _E_cr, 1.0 ); 
  db->getWithDefault( "E_inf", _E_inf, 2.0 ); 
  db->require( "E_st", _E_st ); 
  db->getWithDefault( "C1", _C1, 0.1); 

  if ( _C1 > .20 ){ 
    throw ProblemSetupException( "ERROR: Soot constant C1 is not within published bounds (0-0.2)", __FILE__, __LINE__);
  } 

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void EmpSoot::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "EmpSoot::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &EmpSoot::computeProp, time_substep ); 
  Ghost::GhostType  gn  = Ghost::None;

  _den_label    = VarLabel::find( _den_label_name );
  _T_label      = VarLabel::find( _T_label_name );
  _absorp_label = VarLabel::find( _absorp_label_name );
  _f_label      = VarLabel::find( _f_label_name ); 

  if ( _absorp_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _T_label == 0 ){ 
    throw InvalidValue("Error: Cannot find bsorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _den_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _f_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  }
  
  if ( time_substep == 0 ) {
    
    tsk->computes( _absorp_label ); 

    tsk->requires( Task::OldDW, _T_label,   gn, 0 ); 
    tsk->requires( Task::OldDW, _den_label, gn, 0 ); 
    tsk->requires( Task::OldDW, _f_label,   gn, 0 ); 
    
  } else {

    tsk->modifies( _absorp_label ); 

    tsk->requires( Task::NewDW, _T_label,   gn, 0 ); 
    tsk->requires( Task::NewDW, _den_label, gn, 0 ); 
    tsk->requires( Task::NewDW, _f_label,   gn, 0 ); 

  }

  tsk->modifies( _prop_label ); 
  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void EmpSoot::computeProp(const ProcessorGroup* pc, 
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
    constCCVariable<double> f; 
    Ghost::GhostType  gn  = Ghost::None;

    new_dw->getModifiable( soot_vf,     _prop_label,   matlIndex, patch ); 
    if ( time_substep != 0 ){
    
      new_dw->getModifiable( absorp_coef, _absorp_label, matlIndex, patch ); 

      new_dw->get( temperature, _T_label,   matlIndex, patch, gn, 0 ); 
      new_dw->get( density,     _den_label, matlIndex, patch, gn, 0 ); 
      new_dw->get( f,           _f_label,   matlIndex, patch, gn, 0 ); 

    } else {
      
      new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
      soot_vf.initialize(0.0); 
      absorp_coef.initialize(0.0); 

      old_dw->get( temperature, _T_label,   matlIndex, patch, gn, 0 ); 
      old_dw->get( density,     _den_label, matlIndex, patch, gn, 0 ); 
      old_dw->get( f,           _f_label,   matlIndex, patch, gn, 0 ); 

    }

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double bc = get_carbon_content( f[c] ) * density[c];

      double E = f[c] / ( 1.0 - f[c] ); 
      E = E / _E_st; 

      double C2 = std::max( 0.0, std::min(E-_E_cr, _E_inf - _E_cr ) ); 
      C2 = C2 / ( _E_inf - _E_cr ); 

      if ( temperature[c] > 1000.0 ) { 

        soot_vf[c] = _C1 * C2 * bc * _cmw / _rho_soot; 

      } else { 

        soot_vf[c] = 0.0;

      } 
      
      absorp_coef[c] = 0.01 + std::min( 0.5, (4.0/_opl)*log( 1.0 + 
                       350.0 * soot_vf[c] * temperature[c] * _opl));

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void EmpSoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "EmpSoot::initialize"; 

  _den_label    = VarLabel::find( _den_label_name );
  _T_label      = VarLabel::find( _T_label_name );
  _absorp_label = VarLabel::find( _absorp_label_name );
  _f_label      = VarLabel::find( _f_label_name ); 

  if ( _absorp_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _T_label == 0 ){ 
    throw InvalidValue("Error: Cannot find bsorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _den_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  } else if ( _f_label == 0 ){ 
    throw InvalidValue("Error: Cannot find absorp label in the emperical soot function with name: "+_absorp_label_name,__FILE__,__LINE__);
  }

  Task* tsk = scinew Task(taskname, this, &EmpSoot::initialize);
  tsk->computes(_prop_label); 
  tsk->computes(_absorp_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void EmpSoot::initialize( const ProcessorGroup* pc, 
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
    CCVariable<double> absorp_coef; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 

    new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
    absorp_coef.initialize(0.0); 

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

  }
}
