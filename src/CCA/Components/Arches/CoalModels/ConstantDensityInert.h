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
  * The user cannot specify particle size, particle mass, and particle density.
  * This class does not allow the user to specify the value of density.
  * The reason for this is becuase the ConstantDensityCoal model does not
  * allow the user to specify the value of density, and this class is meant to
  * exercise the same mechaism as the ConstantDensityCoal class.
  * 
  * In the case of particles entering from a boundary, the domain is initialized
  * with very small numbers of particles, so the domain may be initialized
  * with the correct density.
  *
  * Eventually this will not be stored in "CoalModels" directory but will instead
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

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */ 
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

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

  double d_density;                       ///< (Constant) value of particle density
  vector<const VarLabel*> d_massLabels;   ///< Vector of VarLabels for all mass internal coordinates (used to grab the mass source terms, which are used to calculate the length model term \f$ G_{L} \f$ )
  vector<double> d_massScalingConstants;  ///< Vector of scaling constants for all mass internal coordinates (used to scale the mass source terms)

  const VarLabel* d_length_label;         ///< Variable label for particle length internal coordinate
  const VarLabel* d_particle_mass_label;  ///< Variable label for particle mass internal coordinate

  double d_length_low;                    ///< Low clip value for length (if applicable)
  double d_length_hi;                     ///< High clip value for length (if applicable)
  double d_length_scaling_constant;       ///< Scaling constant for particle length internal coordinate
  double d_mass_scaling_constant;         ///< Scaling constant for particle mass internal coordinate

  bool d_doLengthLowClip;  ///< Boolean: do low clipping for length?
  bool d_doLengthHighClip; ///< Boolean: do high clipping for length?
  bool d_useLength;        ///< Boolean: use particle length internal coordinate?
  bool d_useMass;          ///< Boolean: use particle mass internal coordinate?

}; // end ConstantDensityInert
} // end namespace Uintah
#endif

