#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string srcName, SimulationStateP& shared_state,
                                vector<std::string> required_labels ) : 
d_srcName(srcName), d_sharedState( shared_state ), d_requiredLabels(required_labels)
{
  // Create source term labels in child classes
  // (some have different types, e.g. CCVariable<Vector>)
  d_labelSchedInit  = false; 
  _init_type = "constant"; 
}

SourceTermBase::SourceTermBase( std::string srcName, 
                                SimulationStateP& shared_state,
                                vector<std::string> required_labels,
                                ArchesLabel* fieldLabels ) : 
d_srcName(srcName), d_sharedState( shared_state ), d_requiredLabels(required_labels), d_fieldLabels(fieldLabels)
{
  d_labelSchedInit = false;
  _init_type = "constant";
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(d_srcLabel); 
}
