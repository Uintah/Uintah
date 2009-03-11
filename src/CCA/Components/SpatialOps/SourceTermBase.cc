
#include <CCA/Components/SpatialOps/SourceTermBase.h>
#include <CCA/Components/SpatialOps/SourceTermFactory.h>
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
  d_labelActualInit = false; 
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(d_srcLabel); 
}


