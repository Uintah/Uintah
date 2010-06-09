#ifndef Uintah_Component_Arches_CharOxidation_h
#define Uintah_Component_Arches_CharOxidation_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>

//===========================================================================

/**
  * @class    CharOxidation
  * @author   Charles Reid
  * @date     June 2010 
  *
  * @brief    A parent class for char oxidation models
  *
  */

namespace Uintah{

class CharOxidation: public ModelBase {
public: 

  CharOxidation( std::string modelName, 
                 SimulationStateP& shared_state, 
                 const ArchesLabel* fieldLabels,
                 vector<std::string> reqICLabelNames, 
                 vector<std::string> reqScalarLabelNames,
                 int qn );

  virtual ~CharOxidation();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Grab model-independent devolatilization parameters */
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for CharOxidation parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for CharOxidation parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation methods

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("CharOxidation") */
  inline string getType() {
    return "CharOxidation"; }


protected:

  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_particle_temperature_label;
  const VarLabel* d_weight_label;

  double d_rc_scaling_factor;   ///< Scaling factor for raw coal internal coordinate
  double d_char_scaling_factor; ///< Scaling factor for char mass internal coordinate
  double d_pt_scaling_factor;   ///< Scaling factor for particle temperature internal coordinate
  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

}; // end CharOxidation
} // end namespace Uintah
#endif
