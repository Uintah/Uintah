#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, 
                      MaterialManagerP& materialManager,
                      ArchesLabel* fieldLabels,
                      vector<std::string> reqICLabelNames, 
                      vector<std::string> reqScalarLabelNames,
                      int qn ) : 
            d_modelName(modelName),  d_materialManager( materialManager ), d_fieldLabels(fieldLabels), 
            d_icLabels(reqICLabelNames), d_scalarLabels(reqScalarLabelNames), d_quadNode(qn)
{
  // The type and number of d_modelLabel and d_gasLabel
  // is model-dependent, so the creation of these labels 
  // go in the model class constructor.
  // (Note that the labels themselves are still defined in 
  //  the parent class...)
 
  d_labelSchedInit  = false; 
}

ModelBase::~ModelBase()
{
  VarLabel::destroy(d_modelLabel); 
  VarLabel::destroy(d_gasLabel); 
}
