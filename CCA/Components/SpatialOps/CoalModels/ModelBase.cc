
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, SimulationStateP& sharedState,
                      const Fields* fieldLabels
                      vector<std::string> icLabelNames, int qn ) : 
d_modelName(modelName), d_sharedState( sharedState ), d_icLabels(icLabelNames), d_quadNode(qn)
{
  //Create a label for this source term. 
  d_modelLabel = VarLabel::create(modelName, CCVariable<double>::getTypeDescription()); 

  d_labelSchedInit  = false; 
}

ModelBase::~ModelBase()
{
  VarLabel::destroy(d_modelLabel); 
}


