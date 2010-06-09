#ifndef Uintah_Component_Arches_ConstantSize_h
#define Uintah_Component_Arches_ConstantSize_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    ConstantSize
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    Calculates the density based on the assumption of a constant
  *           particle size.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ArchesLabel;
class ConstantSizeBuilder: public ModelBuilder {
public: 
  ConstantSizeBuilder( const std::string          & modelName,
                       const vector<std::string>  & reqICLabelNames,
                       const vector<std::string>  & reqScalarLabelNames,
                       const ArchesLabel          * fieldLabels,
                       SimulationStateP           & sharedState,
                       int qn );

  ~ConstantSizeBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ConstantSize: public ParticleDensity {
public: 

  ConstantSize( std::string modelName, 
                SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames,
                int qn );

  ~ConstantSize();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  // No initVars() method because no special variables needed

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
                     DataWarehouse* new_dw );

  /** @brief  Schedule the calculation of the density */
  void sched_computeParticleDensity( const LevelP& level,
                                     SchedulerP& sched,
                                     int timeSubStep );

  /** @brief  Actually compute the density */
  void computeParticleDensity( const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw );

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

  const VarLabel* d_length_label;
  const VarLabel* d_raw_coal_mass_label;
  //const VarLabel* d_char_mass_label;
  //const VarLabel* d_moisture_mass_label;
  const VarLabel* d_weight_label;

  double d_length_scaling_factor; ///< Scaling factor for particle length internal coordinate
  double d_rc_scaling_factor;     ///< Scaling factor for raw coal internal coordinate
  //double d_char_scaling_factor;
  //double d_moisture_scaling_factor;

}; // end ConstantSize
} // end namespace Uintah
#endif

