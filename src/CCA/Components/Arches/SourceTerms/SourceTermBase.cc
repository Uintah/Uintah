#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string srcName, SimulationStateP& sharedState,
                        vector<std::string> reqLabelNames ) : 
d_srcName(srcName), d_sharedState( sharedState ), d_requiredLabels(reqLabelNames)
{
  // Create source term labels in child classes
  // (some have different types, e.g. CCVariable<Vector>)
  d_labelSchedInit  = false; 
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(d_srcLabel); 
}
