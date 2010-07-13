#ifndef Uintah_Component_Arches_ConstantModel_h
#define Uintah_Component_Arches_ConstantModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

namespace Uintah{

//===========================================================================

/**
  * @class    ConstantModel
  * @author   Jeremy Thornock, Charles Reid
  * @date     July 2010: Cleaning up
  *
  * @brief    A simple constant "growth" model; G = constant.
  *
  * @details
  * The ConstantModel class is a simple constant model. 
  * The user provides a constant that is the value for the UNSCALED internal coordinate.
  * The ConstantModel class then adjusts it to be for the SCALED internal coordinate,
  * since that is the way it must be treated to be added properly to the RHS.
  *
  */

//---------------------------------------------------------------------------
// Builder
class ConstantModelBuilder: public ModelBuilder
{
public: 
  ConstantModelBuilder( const std::string          & modelName, 
                        const vector<std::string>  & reqICLabelNames,
                        const vector<std::string>  & reqScalarLabelNames,
                        const ArchesLabel          * fieldLabels,
                        SimulationStateP           & sharedState,
                        int qn );
  ~ConstantModelBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ConstantModel: public ModelBase {
public: 

  ConstantModel( std::string modelName, 
                 SimulationStateP& shared_state, 
                 const ArchesLabel* fieldLabels,
                 vector<std::string> reqICLabelNames, 
                 vector<std::string> reqScalarLabelNames,
                 int qn );

  ~ConstantModel();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief Interface for the inputfile and set constants */ 
  virtual void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Actually do dummy solve (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /////////////////////////////////////////////////
  // Model computation methods

  /** @brief Schedule the calculation of the source term */ 
  virtual void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );

  /** @brief Actually compute the source term */ 
  virtual void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  ///////////////////////////////////////////////
  // Access methods

  inline string getType() {
    return "Constant"; }


protected:

  const VarLabel* d_ic_label;     ///< The internal coordinate label for the internal coordinate to which the constant model is being applied
  const VarLabel* d_weight_label; ///< The DQMOM weight label

  double d_constant;             ///< The model term (G) value
  double d_ic_scaling_constant;  ///< Scaling constant for the internal coordinate to which the constant model is being applied
  double d_w_scaling_constant;   ///< Scaling constant for the DQMOM weight
  double d_w_small;              ///< Value at which weights are considered approx. 0
  
  double d_low;      ///< Low clip value for length (if applicable)
  double d_high;     ///< High clip value for length (if applicable)

  bool d_doLowClip;  ///< Boolean: do low clipping for length?
  bool d_doHighClip; ///< Boolean: do high clipping for length?

  bool d_reachedLowClip;  ///< Boolean: has the low clip been reached?
  bool d_reachedHighClip; ///< Boolean: has the high clip been reached?

}; // end ConstantModel
} // end namespace Uintah
#endif

