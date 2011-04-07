#ifndef Uintah_Component_Arches_Devolatilization_h
#define Uintah_Component_Arches_Devolatilization_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>

namespace Uintah{

//===========================================================================

/**
  * @class    Devolatilization
  * @author   Charles Reid
  * @date     October 2009
  *
  * @brief    A parent class for devolatilization models.
  *
  */

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
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief Schedule the dummy initialization required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation methods

  /** @brief  Get raw coal reaction rate (see Glacier) */
  virtual double calcRawCoalReactionRate() = 0;

  /** @brief  Get gas volatile production rate (see Glacier) */
  virtual double calcGasDevolRate() = 0;

  /** @brief  Get char production rate (see Glacier) */
  virtual double calcCharProductionRate() = 0;

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("Devolatilization") */
  inline string getType() {
    return "Devolatilization"; }

protected:

  const VarLabel* d_raw_coal_mass_label;        ///< Variable label for raw coal mass internal coordinate
  const VarLabel* d_char_mass_label;            ///< Variable label for char mass internal coordinate 
  const VarLabel* d_moisture_mass_label;        ///< Variable label for moisture mass internal coordinate  
  const VarLabel* d_particle_temperature_label; ///< Variable label for particle temperature internal coordinate   
  const VarLabel* d_gas_temperature_label;      ///< Variable label for gas temperature internal coordinate    
  const VarLabel* d_weight_label;               ///< Variable label for weight internal coordinate 

  double d_rc_scaling_constant;                   ///< Scaling factor for raw coal internal coordinate
  double d_char_scaling_constant;                 ///< Scaling factor for char mass internal coordinate
  double d_pt_scaling_constant;                   ///< Scaling factor for particle temperature internal coordinate

  double d_w_scaling_constant;                    ///< Scaling factor for weight
  double d_w_small;                             ///< "small" clip value for weights (model is not calculated if weight value is < d_w_small)

}; // end Devolatilization
} // end namespace Uintah
#endif
