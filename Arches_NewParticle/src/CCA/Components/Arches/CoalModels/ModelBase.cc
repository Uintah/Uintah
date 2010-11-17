#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, 
                      SimulationStateP& sharedState,
                      const ArchesLabel* fieldLabels,
                      vector<std::string> reqICLabelNames, 
                      vector<std::string> reqScalarLabelNames,
                      int qn ) : 
                      d_modelName(modelName),  d_sharedState( sharedState ), d_fieldLabels(fieldLabels), 
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

// Constructor/destructor for parent class is also called 
// from the constructor/destructor of child class.
// 
// Functions defined here will be overridden if redefined in a child class.
// The child class can explicitly call the parent class method, like this:
// ModelBase::some_function();
//
// Functions declared as pure virtual functions MUST be redefined in child class.
//
// Functions not redefined in child class will use the ModelBase version.


