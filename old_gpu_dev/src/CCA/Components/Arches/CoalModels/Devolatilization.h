#ifndef Uintah_Component_Arches_Devolatilization_h
#define Uintah_Component_Arches_Devolatilization_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    Devolatilization
  * @author   Charles Reid
  * @date     October 2009
  *
  * @brief    A parent class for devolatilization models
  *
  */

namespace Uintah{

class Devolatilization: public ModelBase {
public: 

  Devolatilization( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  virtual ~Devolatilization();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Grab model-independent devolatilization parameters */
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for Devolatilization parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for Devolatilization parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );


  /** @brief  Return a string containing the model type ("Devolatilization") */
  inline string getType() {
    return "Devolatilization"; }

  /** @brief  Return the VarLabel for the model term for char */
  inline const VarLabel* getCharSourceLabel() {
    return d_charLabel; };

protected:

  const VarLabel* d_charLabel;
  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

}; // end Devolatilization
} // end namespace Uintah
#endif
