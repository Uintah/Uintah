#ifndef Uintah_Component_Arches_MaximumTemperature_h
#define Uintah_Component_Arches_MaximumTemperature_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    MaximumTemperature
  * @author   Jeremy Thornock, Julien Pedel, Charles Reid
  * @date     May 2009        Check-in of initial version
  *           November 2009   Verification
  *
  * @brief    A class for calculating the DQMOM model term for the 
  *           Kobayashi-Sarofim coal devolatilization model.
  *
  * The Builder is required because of the Model Factory; the Factory needs
  * some way to create the model term and register it.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class MaximumTemperatureBuilder: public ModelBuilder 
{
public: 
  MaximumTemperatureBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                SimulationStateP                & sharedState,
                                int qn );

  ~MaximumTemperatureBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class MaximumTemperature: public ModelBase {
public: 

  MaximumTemperature( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~MaximumTemperature();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of some special/local variables */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local variables */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

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
                     const int timeSubStep );
  
  inline std::string getType() {
    return "Constant"; }

private:

  const VarLabel* _max_pT_varlabel;
  const VarLabel* _max_pT_weighted_scaled_varlabel;
  const VarLabel* _RHS_source_varlabel;
  const VarLabel* _weight_scaled_varlabel;
  const VarLabel* _particle_temperature_varlabel;

  
  double _max_pT_scaling_constant;   ///< Scaling factor for raw coal internal coordinate
  double _weight_scaling_constant;   ///< Scaling factor for weight 
  double _weight_small;   ///< small weight 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
