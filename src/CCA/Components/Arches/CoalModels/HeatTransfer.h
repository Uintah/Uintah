#ifndef Uintah_Component_Arches_HeatTransfer_h
#define Uintah_Component_Arches_HeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    HeatTransfer
  * @author   Charles Reid
  * @date     November 2009
  *
  * @brief    A heat transfer model parent class .
  *
  */

namespace Uintah{

class HeatTransfer: public ModelBase {
public: 

  HeatTransfer( std::string modelName, 
                SimulationStateP& shared_state, 
                ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
                int qn );

  ~HeatTransfer();

  ////////////////////////////////////////////////////
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

  /** @brief  Get the particle heating rate */
  virtual double calcParticleHeatingRate() = 0;

  /** @brief  Get the gas heating rate */
  virtual double calcGasHeatingRate() = 0;

  /** @brief  Get the particle temperature (see Glacier) */
  virtual double calcParticleTemperature() = 0;

  /** @brief  Get the particle heat capacity (see Glacier) */
  virtual double calcParticleHeatCapacity() = 0;

  /** @brief  Get the convective heat transfer coefficient (see Glacier) */
  virtual double calcConvectiveHeatXferCoeff() = 0;

  /** @brief  Calculate enthalpy of coal off-gas (see Glacier) */
  virtual double calcEnthalpyCoalOffGas() = 0;

  /** @brief  Calculate enthalpy change of the particle (see Glacier) */
  virtual double calcEnthalpyChangeParticle() = 0;

  //////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("HeatTransfer") */
  inline string getType() {
    return "HeatTransfer"; }

  /** @brief  Access function for radiation flag (on/off) */
  inline bool getRadiationFlag(){
    return b_radiation; };   

  /** @brief  Access function for thermal conductivity of particles */
  inline const VarLabel* getabskp(){
    return d_abskp; };  
  

protected:

  bool b_radiation;               ///< Boolean: do radiation calculations?
  
  double pi;
  double d_Pr;                    ///< Prandtl number 
  double d_visc;                  ///< Viscosity of gas
  double d_sigma;                 ///< [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)
  double d_blow;                  ///< Blowing parameter

  const VarLabel* d_weight_label;               ///< Variable label for weights
  const VarLabel* d_length_label;               ///< Label for particle length
  const VarLabel* d_particle_temperature_label; ///< Label for particle temperature
  const VarLabel* d_gas_temperature_label;      ///< Label for gas temperature
  const VarLabel* d_abskp;                      ///< Label for thermal conductivity of the particles

  double d_w_small;                   ///< "small" clip value for weights; if weights are < d_w_small, no model value is computed 
  double d_w_scaling_constant;        ///< Scaling constant for weight
  double d_length_scaling_constant;   ///< Scaling factor for particle size (length)
  double d_pt_scaling_constant;       ///< Scaling factor for particle temperature variable 

  bool d_useLength;    ///< Boolean: is length a scalar/DQMOM variable?
  bool d_useTp;        ///< Boolean: is particle temperature a DQMOM variable?
  bool d_useTgas;      ///< Boolean: is gas temperature a scalar variable?

  // Constant value (if user specifies value of length should be constant)
  double d_length_constant_value;

  // Constant bool
  bool d_constantLength; ///< Boolean: is the length a constant fixed value? (as opposed to an internal coordinate)

}; // end HeatTransfer
} // end namespace Uintah
#endif
