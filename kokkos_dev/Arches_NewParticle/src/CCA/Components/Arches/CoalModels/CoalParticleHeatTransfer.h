#ifndef Uintah_Component_Arches_CoalParticleHeatTransfer_h
#define Uintah_Component_Arches_CoalParticleHeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

namespace Uintah{

/**
  * @class    CoalParticleHeatTransfer
  * @author   Julien Pedel, Jeremy Thornock, Charles Reid
  * @date     November 2009 : Initial version \n
  *           June 2010 : Cleanup/rename
  *
  * @brief    A heat transfer model for coal particles.
  *
  * @details
  * This class requires coal-specific internal coordinates/information.
  * For heat transfer to "inert" (plain) particles, define a different class.
  *
  */

//---------------------------------------------------------------------------
// Builder

class CoalParticleHeatTransferBuilder: public ModelBuilder
{
public: 
  CoalParticleHeatTransferBuilder( const std::string          & modelName,
                             const vector<std::string>  & reqICLabelNames,
                             const vector<std::string>  & reqScalarLabelNames,
                             ArchesLabel          * fieldLabels,
                             SimulationStateP           & sharedState,
                             int qn );

  ~CoalParticleHeatTransferBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class CoalParticleHeatTransfer: public HeatTransfer {
public: 

  CoalParticleHeatTransfer( std::string modelName, 
                SimulationStateP& shared_state, 
                ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
                int qn );

  ~CoalParticleHeatTransfer();

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

  /** @brief  Funtion for calculation of heat capacity (from Merrick) */
  double g1(double z);

  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman, p. 505)
      @param  Tg    Gas temperature
      @param  Tp    Particle temperature */
  double props(double Tg, double Tp);

  /** @brief  Calculate the heat capacity of raw coal
      @param  Tp    Particle temperature */
  double calc_Cp_rawcoal(double Tp);
  
  /** @brief  Calculate the heat capacity of ash 
      @param  Tp    Particle temperature */
  double calc_Cp_ash(double Tp);

  /** @brief  Calculate the heat capacity of char
      @param  Tp    Particle temperature */
  double calc_Cp_char(double Tp);

  vector<double>  d_ash_mass;     ///< Initial ash mass (required)
  vector<double>  d_fixcarb_mass; ///< Initial fixed carbon mass (optional)

  bool d_use_fixcarb_mass;        ///< Boolean: did user specify initial fixed carbon mass?

  double yelem[5];                ///< Mass fractions of each element in coal (C, H, N, O, S respectively)
  double rhop;                    ///< Density of particle 

  const VarLabel* d_raw_coal_mass_label;        ///< Label for raw coal mass
  const VarLabel* d_char_mass_label;            ///< Label for char mass
  const VarLabel* d_moisture_mass_label;        ///< Label for moisture mass
 
  double d_rc_scaling_constant;       ///< Scaling factor for raw coal mass variable
  double d_char_scaling_constant;     ///< Scaling constant for char mass variable 
  double d_moisture_scaling_constant; ///< Scaling constant for moisture mass variable 

  bool d_useRawCoal;   ///< Boolean: is raw coal mass a scalar/DQMOM variable?
  bool d_useChar;      ///< Boolean: is char mass a scalar/DQMOM variable?
  bool d_useMoisture;  ///< Boolean: is moisture mass a scalar/DQMOM variable?

}; // end CoalParticleHeatTransfer
} // end namespace Uintah
#endif
