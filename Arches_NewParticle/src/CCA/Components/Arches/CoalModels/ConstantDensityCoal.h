#ifndef Uintah_Component_Arches_ConstantDensityCoal_h
#define Uintah_Component_Arches_ConstantDensityCoal_h
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
  * @class    ConstantDensityCoal
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    Calculates the coal particle size based on the assumption of a constant
  *           particle density. 
  *
  * @todo
  * For coal particles, we're never interested in specifying a particular value of density \n
  * For that reason, don't need to specify value of density with <density> tag \n
  * ONLY need to specify mass i.c.s and length i.c.
  *
  */

//---------------------------------------------------------------------------
// Builder
class ArchesLabel;
class ConstantDensityCoalBuilder: public ModelBuilder {
public: 
  ConstantDensityCoalBuilder( const std::string          & modelName,
                          const vector<std::string>  & reqICLabelNames,
                          const vector<std::string>  & reqScalarLabelNames,
                          ArchesLabel          * fieldLabels,
                          SimulationStateP           & sharedState,
                          int qn );

  ~ConstantDensityCoalBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ConstantDensityCoal: public ParticleDensity {
public: 

  ConstantDensityCoal( std::string modelName, 
                   SimulationStateP& shared_state, 
                   ArchesLabel* fieldLabels,
                   vector<std::string> reqICLabelNames, 
                   vector<std::string> reqScalarLabelNames,
                   int qn );

  ~ConstantDensityCoal();

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
  
  vector<double> d_ash_mass;             ///< Vector of mass of ash in each environment
  double d_density;                      ///< (Constant) value of density
  vector<const VarLabel*> d_massLabels;  ///< Vector of VarLabels for all mass internal coordinates (used to grab the mass source terms, which are used to calculate the length model term \f$ G_{L} \f$ )
  vector<double> d_massScalingConstants; ///< Vector of scaling constants for mass internal coordinates (used to scale the mass source terms)

  const VarLabel* d_raw_coal_mass_label;  ///< Label for raw coal mass internal coordinate
  const VarLabel* d_char_mass_label;      ///< Label for char mass internal coordinate
  const VarLabel* d_moisture_mass_label;  ///< Label for moisture mass internal coordinate

  double d_rc_scaling_constant;       ///< Scaling constant for raw coal internal coordinate
  double d_char_scaling_constant;     ///< Scaling constant for char internal coordinate
  double d_moisture_scaling_constant; ///< Scaling constant for moisture internal coordinate

  bool d_useRawCoal;   ///< Boolean: use raw coal internal coordinate?
  bool d_useChar;      ///< Boolean: use char internal coordinate?
  bool d_useMoisture;  ///< Boolean: use moisture inernal coordinate?

}; // end ConstantDensityCoal
} // end namespace Uintah
#endif

