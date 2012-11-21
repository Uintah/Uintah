#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace std;
using namespace Uintah; 

PropertyModelBase::PropertyModelBase( std::string prop_name, SimulationStateP& shared_state ) :
  _prop_name( prop_name ), _shared_state( shared_state )
{
  _init_type = "constant"; //Can be overwritten in derived class
  _const_init = 0.0;
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

