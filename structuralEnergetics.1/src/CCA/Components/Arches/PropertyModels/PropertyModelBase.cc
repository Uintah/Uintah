#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/SimulationState.h>

using namespace std;
using namespace Uintah; 

PropertyModelBase::PropertyModelBase( std::string prop_name, SimulationStateP& shared_state ) :
  _prop_name( prop_name ), _shared_state( shared_state )
{
  _init_type = "constant"; //Can be overwritten in derived class
  _const_init = 0.0;
  _prop_type = "not_set"; 
}

PropertyModelBase::~PropertyModelBase()
{
  VarLabel::destroy(_prop_label); 
}

void 
PropertyModelBase::commonProblemSetup( const ProblemSpecP& inputdb )
{

  ProblemSpecP db = inputdb; 

  std::string type; 
  ProblemSpecP db_init = db->findBlock("initialization");
  db_init->getAttribute("type",type); 



  if ( type == "constant" ){ 

    db_init->require("constant",_const_init); 
    _init_type = "constant"; 

  } else if ( type == "gaussian" ){ 

    db_init->require( "amplitude", _a_gauss ); 
    db_init->require( "center", _b_gauss ); 
    db_init->require( "std", _c_gauss ); 

    std::string direction; 
    db_init->require( "direction", direction ); 

    if ( direction == "X" || direction == "x" ){ 
      _dir_gauss = 0; 
    } else if ( direction == "Y" || direction == "y" ){ 
      _dir_gauss = 1; 
    } else if ( direction == "Z" || direction == "z" ){
      _dir_gauss = 2; 
    } 
    db_init->getWithDefault( "shift", _shift_gauss, 0.0 ); 

    _init_type = "gaussian";
  } else if ( type == "physical" ){ 
    _init_type = "physical"; 
  } else { 

    throw ProblemSetupException( "Error: Property model initialization not recognized.", __FILE__, __LINE__);

  } 
}

void 
PropertyModelBase::sched_timeStepInit( const LevelP& level, SchedulerP& sched )
{
  Task* tsk = scinew Task( "PropertyModelBase::timeStepInit", this, &PropertyModelBase::timeStepInit); 

  tsk->computes( _prop_label );   // 2nd compute for Julien_abskp
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 );
  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}

void 
PropertyModelBase::timeStepInit( const ProcessorGroup* pc, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw )
{
//  new_dw->transferFrom(old_dw,_prop_label ,  patches, matls);  // This changes OldDW when used improperly

  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> property_old; 
    old_dw->get(property_old, _prop_label, matlIndex, patch, Ghost::None, 0 );

    CCVariable<double> property_new; 
    new_dw->allocateAndPut( property_new, _prop_label, matlIndex, patch );
    property_new.copyData(property_old);
  }
}
