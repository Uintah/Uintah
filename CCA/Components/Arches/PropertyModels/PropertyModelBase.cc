#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

PropertyModelBase::PropertyModelBase( std::string prop_name, SimulationStateP& shared_state ) :
  _prop_name( prop_name ), _shared_state( shared_state )
{
  _init_type = "constant"; //Can be overwritten in derived class
  _const_init = 1.0;
}

PropertyModelBase::~PropertyModelBase()
{
  VarLabel::destroy(_prop_label); 
}

