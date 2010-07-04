#ifndef Uintah_Component_Arches_ConstantSizeCoal_h
#define Uintah_Component_Arches_ConstantSizeCoal_h
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
  * @class    ConstantSizeCoal
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
  * The mass of the particle currently consists of the sum of the 
  * raw coal mass, char mass, moisture mass, and ash mass.
  * 
  * The d_useChar and d_useMoisture booleans are necessary because these 
  * internal coordinates are (currently) optional.  They do create a lot
  * of "if-else" statements, however, and it would be a good idea to 
  * remove them (eventually).
  */

//---------------------------------------------------------------------------
// Builder

class ArchesLabel;
class ConstantSizeCoalBuilder: public ModelBuilder {
public: 
  ConstantSizeCoalBuilder( const std::string          & modelName,
                           const vector<std::string>  & reqICLabelNames,
                           const vector<std::string>  & reqScalarLabelNames,
                           const ArchesLabel          * fieldLabels,
                           SimulationStateP           & sharedState,
                           int qn );

  ~ConstantSizeCoalBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ConstantSizeCoal: public ParticleDensity {
public: 

  ConstantSizeCoal( std::string modelName, 
                    SimulationStateP& shared_state, 
                    const ArchesLabel* fieldLabels,
                    vector<std::string> reqICLabelNames, 
                    vector<std::string> reqScalarLabelNames,
                    int qn );

  ~ConstantSizeCoal();

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

  vector<double> d_ash_mass;    ///< Vector of mass of ash in each environment

  const VarLabel* d_length_label;
  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_moisture_mass_label;

  double d_length_scaling_constant;   ///< Scaling constant for particle length internal coordinate
  double d_rc_scaling_constant;       ///< Scaling constant for raw coal internal coordinate
  double d_char_scaling_constant;     ///< Scaling constant for char internal coordinate
  double d_moisture_scaling_constant; ///< Scaling constant for moisture internal coordinate

  bool d_useLength;    ///< Boolean: use length internal coordinate?
  bool d_useRawCoal;   ///< Boolean: use raw coal internal coordinate?
  bool d_useChar;      ///< Boolean: use char internal coordinate?
  bool d_useMoisture;  ///< Boolean: use moisture inernal coordinate?

}; // end ConstantSizeCoal
} // end namespace Uintah
#endif

