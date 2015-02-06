#ifndef Uintah_Component_Arches_CharOxidation_h
#define Uintah_Component_Arches_CharOxidation_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    CharOxidation
  * @author   Julien Pedel
  * @date     Feb 2011
  *
  * @brief    A parent class for CharOxidation models
  *
  */

namespace Uintah{

class CharOxidation: public ModelBase {
public: 

  CharOxidation( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  virtual ~CharOxidation();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Grab model-independent CharOxidation parameters */
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for CharOxidation parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for CharOxidation parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("CharOxidation") */
  inline string getType() {
    return "CharOxidation"; }

  /** @brief  Return the VarLabel for the model term for char */
  inline const VarLabel* getParticleTempSourceLabel() {
    return d_particletempLabel; };

  inline const VarLabel* getSurfaceRateLabel() {
    return d_surfacerateLabel; };

  inline const VarLabel* getPO2surfLabel() {
    return d_PO2surfLabel; };

protected:

  const VarLabel* d_particletempLabel;
  const VarLabel* d_surfacerateLabel;
  const VarLabel* d_PO2surfLabel;
  double d_w_scaling_constant; 
  double d_w_small; // "small" clip value for zero weights

}; // end CharOxidation
} // end namespace Uintah
#endif
