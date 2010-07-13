#ifndef Uintah_Component_Arches_ConstantSizeInert_h
#define Uintah_Component_Arches_ConstantSizeInert_h
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
  * @class    ConstantSizeInert
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    Calculates the density of a coal particle based on the assumption 
  *           of a constant particle size.
  *
  * @details
  * The density can be calculated based on the mass of the particle and 
  * the assumption of a constant particle size.
  * 
  * The mass of the particle is an internal coordinate.
  *
  * Eventually this will not be stored in the "CoalModels/" directory, but rather 
  * will go into a directory for more generic two-phase models.
  * 
  */

//---------------------------------------------------------------------------
// Builder

class ArchesLabel;
class ConstantSizeInertBuilder: public ModelBuilder {
public: 
  ConstantSizeInertBuilder( const std::string          & modelName,
                           const vector<std::string>  & reqICLabelNames,
                           const vector<std::string>  & reqScalarLabelNames,
                           const ArchesLabel          * fieldLabels,
                           SimulationStateP           & sharedState,
                           int qn );

  ~ConstantSizeInertBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ConstantSizeInert: public ParticleDensity {
public: 

  ConstantSizeInert( std::string modelName, 
                    SimulationStateP& shared_state, 
                    const ArchesLabel* fieldLabels,
                    vector<std::string> reqICLabelNames, 
                    vector<std::string> reqScalarLabelNames,
                    int qn );

  ~ConstantSizeInert();

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

  ///////////////////////////////////////////////////
  // Model calculation methods

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

  vector<double> d_ash_mass;              ///< Vector of mass of ash in each environment

  const VarLabel* d_length_label;         ///< Variable label for particle length internal coordinate
  const VarLabel* d_particle_mass_label;  ///< Variable label for particle mass internal coordinate

  double d_length_scaling_constant;       ///< Scaling constant for particle length internal coordinate
  double d_mass_scaling_constant;         ///< Scaling constant for particle mass internal coordinate

  bool d_useLength;    ///< Boolean: use length internal coordinate?
  bool d_useMass;      ///< Boolean: use mass inernal coordinate?

}; // end ConstantSizeInert
} // end namespace Uintah
#endif

