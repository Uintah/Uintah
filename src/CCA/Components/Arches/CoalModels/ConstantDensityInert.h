#ifndef Uintah_Component_Arches_ConstantDensityInert_h
#define Uintah_Component_Arches_ConstantDensityInert_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah{

//===========================================================================

/**
  * @class    ConstantDensityInert
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    Calculates the coal particle size based on the assumption of a constant
  *           particle density. 
  *
  * @details
  * Eventually this will not be stored in "Coalmodels" but will instead
  * go into a directory for more generic two-phase models.
  *
  */

//---------------------------------------------------------------------------
// Builder

class ArchesLabel;
class ConstantDensityInertBuilder: public ModelBuilder {
public: 
  ConstantDensityInertBuilder( const std::string          & modelName,
                          const vector<std::string>  & reqICLabelNames,
                          const vector<std::string>  & reqScalarLabelNames,
                          const ArchesLabel          * fieldLabels,
                          SimulationStateP           & sharedState,
                          int qn );

  ~ConstantDensityInertBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ConstantDensityInert: public ParticleDensity {
public: 

  ConstantDensityInert( std::string modelName, 
                   SimulationStateP& shared_state, 
                   const ArchesLabel* fieldLabels,
                   vector<std::string> reqICLabelNames, 
                   vector<std::string> reqScalarLabelNames,
                   int qn );

  ~ConstantDensityInert();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief  Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief  Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  /** @brief  Schedule the calculation of the density */
  void sched_computeParticleDensity( const LevelP& level,
                                     SchedulerP& sched,
                                     int timeSubStep );

  /** @brief  Actually compute the density */
  void computeParticleDensity( const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               int timeSubStep );

  double calcSize() {
    return 0.0; };

  double calcArea() {
    return 0.0; };

  double calcParticleDensity() {
    return 0.0; };

  ///////////////////////////////////////////////////
  // Get/set methods

  /* getType method is defined in parent class... */

private:

  double d_density;
  vector<const VarLabel*> massLabels; ///< Vector of VarLabels for all mass internal coordinates (used to grab the mass source terms, which are used to calculate the length model term \f$ G_{L} \f$ )

  const VarLabel* d_length_label;
  const VarLabel* d_particle_mass_label;

  double d_length_scaling_constant;
  double d_mass_scaling_constant;

  bool d_useLength;
  bool d_useMass;

}; // end ConstantDensityInert
} // end namespace Uintah
#endif

