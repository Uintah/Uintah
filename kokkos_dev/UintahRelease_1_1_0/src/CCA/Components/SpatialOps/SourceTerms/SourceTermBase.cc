
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermBase.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string srcName, SimulationStateP& sharedState,
                        vector<std::string> reqLabelNames ) : 
d_srcName(srcName), d_sharedState( sharedState ), d_requiredLabels(reqLabelNames)
{
  //Create a label for this source term. 
  d_srcLabel = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 

  d_labelSchedInit  = false; 
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(d_srcLabel); 
}


