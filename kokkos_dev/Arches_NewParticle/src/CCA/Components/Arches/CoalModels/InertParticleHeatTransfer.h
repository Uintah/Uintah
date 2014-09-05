#ifndef Uintah_Component_Arches_InertParticleHeatTransfer_h
#define Uintah_Component_Arches_InertParticleHeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah{

/**
  * @class    InertParticleHeatTransfer
  * @author   Charles Reid
  * @date     June 2010
  *
  * @brief    A heat transfer model for inert particles.
  *
  * @details
  * 
  * The InertParticleHeatTransfer class describes heat transfer for inert,
  * non-reacting, single-component particles. This is in contrast to the
  * CoalParticleHeatTransfer class, which should be applied exclusively
  * to coal particles.
  *
  * This class is a child of the HeatTransfer class.
  * 
  * This class only requires the mass and heat capacity of a particle.
  *
  * The heat capcity may be an arbitrary function, and is 
  * (initially, for now) assumed constant.
  * 
  * Rather than requiring the mass of particular components, this class
  * only requires the total mass of the particle (as either a normal scalar variable
  * or as a DQMOM scalar variable).
  *
  */

//---------------------------------------------------------------------------
// Builder

class InertParticleHeatTransferBuilder: public ModelBuilder
{
public: 
  InertParticleHeatTransferBuilder( const std::string          & modelName,
                                    const vector<std::string>  & reqICLabelNames,
                                    const vector<std::string>  & reqScalarLabelNames,
                                    ArchesLabel          * fieldLabels,
                                    SimulationStateP           & sharedState,
                                    int qn );

  ~InertParticleHeatTransferBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class InertParticleHeatTransfer: public HeatTransfer {
public: 

  InertParticleHeatTransfer( std::string modelName, 
                             SimulationStateP& shared_state, 
                             ArchesLabel* fieldLabels,
                             vector<std::string> reqICLabelNames, 
                             vector<std::string> reqScalarLabelNames, 
                             int qn );

  ~InertParticleHeatTransfer();

  /////////////////////////////////////////
  // Initialization methods

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

  /////////////////////////////////////////////
  // Model computation methods

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  // TODO: add Glacier computation methods

  /** @brief  Get the particle heating rate (see Glacier) */
  double calcParticleHeatingRate() {
    return 0; }

  /** @brief  Get the gas heating rate (see Glacier) */
  double calcGasHeatingRate() {
    return 0; }

  /** @brief  Get the particle temperature (see Glacier) */
  double calcParticleTemperature() {
    return 0; }

  /** @brief  Get the particle heat capacity (see Glacier) */
  double calcParticleHeatCapacity() {
    return 0; }

  /** @brief  Get the convective heat transfer coefficient (see Glacier) */
  double calcConvectiveHeatXferCoeff() {
    return 0; }

  /** @brief  Calculate enthalpy of coal off-gas (see Glacier) */
  double calcEnthalpyCoalOffGas() {
    return 0; }

  /** @brief  Calculate enthalpy change of the particle (see Glacier) */
  double calcEnthalpyChangeParticle() {
    return 0; }


  /////////////////////////////////////////////////
  // Access methods


private:

  //////////////////////////////////////////////////
  // Private methods for calculation

  /** @brief  Calculate heat capacity of particle (constant, for now) */
  double calc_Cp() {
    return d_Cp; }

  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman, p. 505) 
      @param  Tg  Gas temperature
      @param  Tp  Particle temperature */
  double props(double Tg, double Tp);

  double d_Cp;      ///< Particle heat capacity

  const VarLabel* d_particle_mass_label;        ///< Label for raw coal mass
 
  double d_mass_scaling_constant;     ///< Scaling factor for raw coal mass variable

  bool d_useMass;      ///< Boolean: is particle mass a scalar/DQMOM variable?

}; // end InertParticleHeatTransfer
} // end namespace Uintah

#endif

