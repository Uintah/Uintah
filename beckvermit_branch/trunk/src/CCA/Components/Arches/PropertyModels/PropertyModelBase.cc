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

  } else { 

    throw ProblemSetupException( "Error: Property model initialization not recognized.", __FILE__, __LINE__);

  } 
}

void 
PropertyModelBase::sched_timeStepInit( const LevelP& level, SchedulerP& sched )
{
  Task* tsk = scinew Task( "PropertyModelBase::timeStepInit", this, &PropertyModelBase::timeStepInit); 

  tsk->computes( _prop_label ); 

  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}

void 
PropertyModelBase::timeStepInit( const ProcessorGroup* pc, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw )
{

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> prop; 
    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 

    //value initialization should occur in the property implementation

  }
}
