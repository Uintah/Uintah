
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

ModelBase::ModelBase( std::string modelName, SimulationStateP& sharedState,
                        vector<std::string> reqLabelNames, int qn ) : 
d_modelName(modelName), d_sharedState( sharedState ), d_requiredLabels(reqLabelNames), d_quadNode(qn)
{
  //Create a label for this source term. 
  d_modelLabel = VarLabel::create(modelName, CCVariable<double>::getTypeDescription()); 

  d_labelSchedInit  = false; 
  d_labelActualInit = false; 
}

ModelBase::~ModelBase()
{
  VarLabel::destroy(d_modelLabel); 
}


