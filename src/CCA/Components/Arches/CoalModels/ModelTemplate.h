#ifndef Uintah_Component_Arches_TemplateName_h
#define Uintah_Component_Arches_TemplateName_h
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>

//===========================================================================

/**
  * @class    TemplateName
  * @author   
  * @date     
  *
  * @brief    Insert a brief description here.
  *
  * @detailed 
  * Insert a detailed description here.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ArchesLabel;
class TemplateNameBuilder: public ModelBuilder 
{
public: 
  TemplateNameBuilder( const std::string          & modelName,
                       const vector<std::string>  & reqICLabelNames,
                       const vector<std::string>  & reqScalarLabelNames,
                       ArchesLabel          * fieldLabels,
                       SimulationStateP           & sharedState,
                       int qn );

  ~TemplateNameBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class TemplateName: public ModelBase {
public: 

  TemplateName( std::string modelName, 
                SimulationStateP& shared_state, 
                ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames,
                int qn );

  ~TemplateName();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  ///////////////////////////////////////////////////
  // Get/set methods

  /** @brief  Return a string containing the model type */
  inline string getType() {
    return "TemplateType"; }

private:

  const VarLabel* d_required_internal_coordinate_labels;
  const VarLabel* d_weight_label;

  double d_ic_scaling_factor;
  
  double d_w_scaling_factor;
  double d_w_small;

}; // end TemplateName
} // end namespace Uintah
#endif
