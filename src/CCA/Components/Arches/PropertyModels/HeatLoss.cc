#include <CCA/Components/Arches/PropertyModels/HeatLoss.h>
#include <CCA/Components/Arches/Properties.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
HeatLoss::HeatLoss( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 

  _constant_heat_loss = false; 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
HeatLoss::~HeatLoss( )
{
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void HeatLoss::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  if ( db->findBlock("constant_heat_loss") ){ 
    _constant_heat_loss = true; 
    db->findBlock("constant_heat_loss")->getAttribute("value",_constant);
  } else {  
	  db->require( "enthalpy_label", _enthalpy_label_name ); 
	  db->getWithDefault( "adiabatic_enthalpy_label" , _adiab_h_label_name , "adiabaticenthalpy" );
	  db->getWithDefault( "sensible_enthalpy_label"  , _sen_h_label_name   , "sensibleenthalpy" );


	  _low_hl  = -1; 
	  _high_hl =  1;
	  if ( db->findBlock( "hl_bounds" ) ) { 
	  	db->findBlock( "hl_bounds" )->getAttribute("low"  , _low_hl );
	  	db->findBlock( "hl_bounds" )->getAttribute("high" , _high_hl);
    }

	  _noisy_heat_loss = false; 
	  if ( db->findBlock( "noisy_hl_warning" ) ){ 
	  	_noisy_heat_loss; 
	  } 
  }
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void HeatLoss::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{

  std::string taskname = "HeatLoss::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &HeatLoss::computeProp, time_substep ); 

  if ( !_constant_heat_loss ){ 
	  _enthalpy_label = 0; 
	  _adiab_h_label  = 0; 
	  _sen_h_label    = 0; 

	  _enthalpy_label = VarLabel::find( _enthalpy_label_name ); 
	  _adiab_h_label  = VarLabel::find( _adiab_h_label_name  ); 
	  _sen_h_label    = VarLabel::find( _sen_h_label_name    ); 

	  if ( _enthalpy_label == 0 ){ 
    	throw InvalidValue( "Error: Could not find enthalpy label with name: "+_enthalpy_label_name, __FILE__, __LINE__); 
	  } 
	  if ( _sen_h_label == 0 ){ 
    	throw InvalidValue( "Error: Could not find sensible enthalpy label with name: "+_sen_h_label_name, __FILE__, __LINE__); 
	  } 
	  if ( _adiab_h_label == 0 ){ 
    	throw InvalidValue( "Error: Could not find adiabatic enthalpy label with name: "+_adiab_h_label_name, __FILE__, __LINE__); 
	  } 

	  if ( time_substep == 0 && !_has_been_computed ){ 

	  	tsk->computes( _prop_label ); 

	  	tsk->requires( Task::OldDW , _enthalpy_label , Ghost::None , 0 );
	  	tsk->requires( Task::OldDW , _sen_h_label    , Ghost::None , 0 );
	  	tsk->requires( Task::OldDW , _adiab_h_label  , Ghost::None , 0 );

	  } else { 

	  	tsk->modifies( _prop_label ); 

	  	tsk->requires( Task::NewDW , _enthalpy_label , Ghost::None , 0 );
	  	tsk->requires( Task::NewDW , _sen_h_label    , Ghost::None , 0 );
	  	tsk->requires( Task::NewDW , _adiab_h_label  , Ghost::None , 0 );

	  } 
  } else { 
	  if ( time_substep == 0 && !_has_been_computed ){ 

	  	tsk->computes( _prop_label ); 

	  } else { 

	  	tsk->modifies( _prop_label ); 

	  } 
  }

  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
	_has_been_computed = true; 

}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void HeatLoss::computeProp(const ProcessorGroup* pc, 
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


    CellIterator iter = patch->getCellIterator(); 

		bool oob_up = false; 
		bool oob_dn = false; 
    if ( _constant_heat_loss ){ 

      CCVariable<double> prop; 

		  if ( time_substep == 0 ){ 

        new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 

		  } else { 

        new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 

		  } 

      prop.initialize( _constant ); 

    } else { 

      CCVariable<double> prop; 

		  constCCVariable<double> h;       //enthalpy
		  constCCVariable<double> h_sen;   //sensible enthalpy
		  constCCVariable<double> h_ad;    //adiabatic enthalpy

		  if ( time_substep == 0 ){ 

        new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
        prop.initialize(0.0); 

		  	old_dw->get( h     , _enthalpy_label , matlIndex , patch , Ghost::None , 0 );
		  	old_dw->get( h_sen , _sen_h_label    , matlIndex , patch , Ghost::None , 0 );
		  	old_dw->get( h_ad  , _adiab_h_label  , matlIndex , patch , Ghost::None , 0 );

		  } else { 

        new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 

		  	new_dw->get( h     , _enthalpy_label , matlIndex , patch , Ghost::None , 0 );
		  	new_dw->get( h_sen , _sen_h_label    , matlIndex , patch , Ghost::None , 0 );
		  	new_dw->get( h_ad  , _adiab_h_label  , matlIndex , patch , Ghost::None , 0 );

		  } 

      for (iter.begin(); !iter.done(); iter++){

		  	IntVector c = *iter; 

		  	double small = 1e-16;
        double numerator = h_ad[c] - h[c]; 

        double hl = ( numerator ) / ( h_sen[c] + small ); 

		  	if ( hl < _low_hl ){ 
		  		hl     = -1;
		  		oob_dn = true;
		  	} 
		  	if ( hl > _high_hl ){ 
		  		hl     = 1;
		  		oob_up = true;
		  	} 

		  	prop[c] = hl;

      }
      if ( _noisy_heat_loss ) { 
       
        if ( oob_up || oob_dn ) {  
		  		std::cout << "Patch with bounds: " << patch->getCellLowIndex() << " to " << patch->getCellHighIndex()  << std::endl;
          if ( oob_dn ) 
            std::cout << "   --> lower heat loss exceeded. " << std::endl;
          if ( oob_up ) 
            std::cout << "   --> upper heat loss exceeded. " << std::endl;
        } 
      } 
    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void HeatLoss::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void HeatLoss::dummyInit( const ProcessorGroup* pc, 
                                            const PatchSubset* patches, 
                                            const MaterialSubset* matls, 
                                            DataWarehouse* old_dw, 
                                            DataWarehouse* new_dw )
{
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void HeatLoss::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "HeatLoss::initialize"; 

  Task* tsk = scinew Task(taskname, this, &HeatLoss::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void HeatLoss::initialize( const ProcessorGroup* pc, 
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
