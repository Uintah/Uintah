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
  * @brief    A heat transfer model parent class 
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
  void problemSetup(const ProblemSpecP& db, int qn);
  
  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for HeatTransfer parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for HeatTransfer parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy solve (sched_dummyInit is defined in ModelBase parent class) */
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
    return _radiation; };   

  /** @brief  Return the VarLabel for the model term for absorption coefficient */
  inline const VarLabel* getabskpLabel() {
    return d_abskpLabel; };

  inline const VarLabel* getqconvLabel() {
    return d_qconvLabel; };

  inline const VarLabel* getqradLabel() {
    return d_qradLabel; };

protected:

  bool _radiation;
  const VarLabel* d_abskpLabel;
  const VarLabel* d_qconvLabel;
  const VarLabel* d_qradLabel;
  double d_w_scaling_constant;
  double d_w_small; // "small" clip value for zero weights

}; // end HeatTransfer
} // end namespace Uintah
#endif
