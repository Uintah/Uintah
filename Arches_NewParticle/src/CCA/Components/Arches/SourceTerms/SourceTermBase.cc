#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string srcName, 
                                SimulationStateP& shared_state,
                                vector<std::string> required_labels ) : 
_src_name(srcName), _shared_state( shared_state ), _required_labels(required_labels)
{
  // Create source term labels in child classes
  // (some have different types, e.g. CCVariable<Vector>)
  _label_sched_init  = false; 
  _init_type = "constant"; 
  d_label_init = false;
}

SourceTermBase::SourceTermBase( std::string srcName, 
                                SimulationStateP& shared_state,
                                vector<std::string> required_labels,
                                ArchesLabel* fieldLabels ) : 
_src_name(srcName), _shared_state( shared_state ), _required_labels(required_labels), _fieldLabels(fieldLabels)
{
  _label_sched_init = false;
  _init_type = "constant";
  d_label_init = true;
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(_src_label); 
}
