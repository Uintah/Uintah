#include <CCA/Components/Arches/PropertyModels/EmpSoot.h>

using namespace Uintah; 

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
  
  db->require( "carbon_content", _carb_content );

  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources") ){ 

    // Look for the opl specified in the radiation model: 
    ProblemSpecP sources_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
    for (ProblemSpecP src_db = sources_db->findBlock("src");
          src_db !=0; src_db = src_db->findNextBlock("src")){

      string type; 
      src_db->getAttribute("type", type); 

      if ( type == "do_radiation" ){ 

        src_db->findBlock("DORadiationModel")->require("opl", _opl ); 

      } else { 

        //radiation model (not DO)  doesn't currently set opl
        db->require( "opl", _opl ); 

      } 
    }
  } else { 

    // if no DORadiation model exists, then require user to specify the OPL
    proc0cout << "No DORadiation model specified.  OPL set in the soot model" << endl;
    db->require( "opl", _opl );   

  } 


  db->getWithDefault( "density_label", _den_label_name, "density"); 
  db->getWithDefault( "scaling_factor", _scale_factor, 1.0 ); 
  db->getWithDefault( "absorption_label", _absorp_label_name, "absorpIN"); 
  db->getWithDefault( "temperature_label", _T_label_name, "temperature"); 
  db->getWithDefault( "soot_density", _rho_soot, 1950.0); 
  db->getWithDefault( "c1", _c1, 0.1); 

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void EmpSoot::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "EmpSoot::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &EmpSoot::computeProp, time_substep ); 

  if ( !(_has_been_computed) ) {

    if ( time_substep == 0 ) {
      
      tsk->computes( _prop_label ); 
			tsk->computes( _absorp_label ); 

      tsk->requires( Task::OldDW, _T_label, Ghost::None, 0 ); 
      tsk->requires( Task::OldDW, _den_label, Ghost::None, 0 ); 

    } else {

      tsk->modifies( _prop_label ); 
      tsk->modifies( _absorp_label ); 

      tsk->requires( Task::NewDW, _T_label, Ghost::None, 0 ); 
      tsk->requires( Task::NewDW, _den_label, Ghost::None, 0 ); 

    }

    sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
    
    _has_been_computed = true; 

  }
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

    std::cout << " MATERIAL INDEX = " << matlIndex << std::endl;

    CCVariable<double> soot_vf; 
    CCVariable<double> absorp_coef; 
    constCCVariable<double> temperature; 
    constCCVariable<double> density; 

    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){

      new_dw->getModifiable( soot_vf, _prop_label, matlIndex, patch ); 
      new_dw->getModifiable( absorp_coef, _absorp_label, matlIndex, patch ); 

      new_dw->get( temperature, _T_label, matlIndex, patch, Ghost::None, 0 ); 
      new_dw->get( density, _den_label, matlIndex, patch, Ghost::None, 0 ); 

    } else {

      new_dw->allocateAndPut( soot_vf, _prop_label, matlIndex, patch ); 
      new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
      soot_vf.initialize(0.0); 
      absorp_coef.initialize(0.0); 

      old_dw->get( temperature, _T_label, matlIndex, patch, Ghost::None, 0 ); 
      old_dw->get( density, _T_label, matlIndex, patch, Ghost::None, 0 ); 

    }

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double bc = _carb_content * density[c];


      if ( temperature[c] > 1000.0 ) { 
        soot_vf[c] = _scale_factor * ( _c1 * bc * _cmw ) / _rho_soot; 
      } else { 
        soot_vf[c] = 0.0;
      } 
      
      absorp_coef[c] = 0.01 + std::min( 0.5, (4.0/_opl)*log( 1.0 + 
                       350.0 * soot_vf[c] * temperature[c] * _opl));

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void EmpSoot::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "EmpSoot::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &EmpSoot::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  tsk->computes(_absorp_label); 
  tsk->requires( Task::OldDW, _absorp_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void EmpSoot::dummyInit( const ProcessorGroup* pc, 
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

    CCVariable<double> absorp_coef; 
    constCCVariable<double> old_absorp_coef; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    old_dw->get( old_prop, _prop_label, matlIndex, patch, Ghost::None, 0); 

    new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
    old_dw->get( old_absorp_coef, _absorp_label, matlIndex, patch, Ghost::None, 0); 

    prop.copyData( old_prop );
    absorp_coef.copyData( old_absorp_coef ); 

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
