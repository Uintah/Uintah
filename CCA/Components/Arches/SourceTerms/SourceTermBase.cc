#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string src_name, SimulationStateP& shared_state,
                                vector<std::string> required_labels, std::string type ) : 
_src_name(src_name), _shared_state( shared_state ), _required_labels(required_labels), _type( type )
{
  _init_type = "constant"; 
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(_src_label); 
}
