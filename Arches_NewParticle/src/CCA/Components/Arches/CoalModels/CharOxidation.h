#ifndef Uintah_Component_Arches_CharOxidation_h
#define Uintah_Component_Arches_CharOxidation_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>

namespace Uintah{

//===========================================================================

/**
  * @class    CharOxidation
  * @author   Charles Reid
  * @date     June 2010 
  *
  * @brief    A parent class for char oxidation models
  *
  */

class CharOxidation: public ModelBase {
public: 

  CharOxidation( std::string modelName, 
                 SimulationStateP& shared_state, 
                 const ArchesLabel* fieldLabels,
                 vector<std::string> reqICLabelNames, 
                 vector<std::string> reqScalarLabelNames,
                 int qn );

  virtual ~CharOxidation();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Grab model-independent devolatilization parameters */
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation methods

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("CharOxidation") */
  inline string getType() {
    return "CharOxidation"; }

  /** @brief    Get label for gas mixture fraction source term associated with oxidation of char by O2 */
  inline const VarLabel* getO2GasSourceLabel() {
    return d_O2GasModelLabel; }

  /** @brief    Get label for gas mixture fraction source term associated with oxidation of char by H2 */
  inline const VarLabel* getH2GasSourceLabel() {
    return d_H2GasModelLabel; }

  /** @brief    Get label for gas mixture fraction source term associated with oxidation of char by CO2 */
  inline const VarLabel* getCO2GasSourceLabel() {
    return d_CO2GasModelLabel; }

  /** @brief    Get label for gas mixture fraction source term associated with oxidation of char by H2O */
  inline const VarLabel* getH2OGasSourceLabel() {
    return d_H2OGasModelLabel; }

  /** @brief    Set the TabPropsInterface so char oxidation methods can request species labels */
  inline void setTabPropsInterface( TabPropsInterface* interface ) {
    d_TabPropsInterface = interface; }

  /** @brief    Given an oxidizer species, return the VarLabel for the gas source term associated with that char reaction 
      @param    species_name  Name of oxidizer species (identifies the char reaction) 
      @returns  */
  inline const VarLabel* getGasModelLabel( string species_name ) {
    vector<string>::iterator oxidizerIter = find( oxidizer_name_.begin(), oxidizer_name_.end(), species_name ); //find( OxidizerLabels_.begin(), OxidizerLabels_.end(), species_name);
    vector<const VarLabel*>::iterator gasIter = GasModelLabels_.begin() + std::distance(oxidizer_name_.begin(), oxidizerIter);;
    return (*gasIter);
  }
    
    

protected:

  TabPropsInterface* d_TabPropsInterface; ///< TabProps interface object, used to request specific species from the table

  // ---------------------------
  // Model labels

  // Gas model labels
  const VarLabel*  d_O2GasModelLabel;   ///< Variable label for gas source term due to char oxidation by O2
  const VarLabel*  d_H2GasModelLabel;   ///< Variable label for gas source term due to char oxidation by H2
  const VarLabel* d_CO2GasModelLabel;   ///< Variable label for gas source term due to char oxidation by CO2
  const VarLabel* d_H2OGasModelLabel;   ///< Variable label for gas source term due to char oxidation by H2O

  vector<const VarLabel*> GasModelLabels_;  ///< Vector of variable labels; holds the gas source terms, declared above (if the char oxidation class is extended to arbitrary oxidation reactions, this will allow for more generality)

  vector<double> E_;  ///< Vector of activation energies for char reactions
  vector<double> A_;  ///< Vector of pre-exponential factors for char reactions
  vector<double> Sc_; ///< Vector of Schmidt numbers for oxidizer species (O2, H2, CO2, and H2O, in that order)


  // ---------------------------
  // Gas species labels

  vector<string> oxidizer_name_;    ///< Vector of strings containing the name of oxidizer species for each char reaction
                                    ///  (These are requested from the TabProps interface later, once the table has been read)
  vector<const VarLabel*> OxidizerLabels_; ///< Vector of variable labels for oxidizer concentrations for each char reaction

  // ----------------------------
  // DQMOM labels & constants

  const VarLabel* d_weight_label;       ///< Variable label for DQMOM weight 
  const VarLabel* d_char_mass_label;    ///< Variable label for char mass internal coordinate
  const VarLabel* d_length_label;       ///< Variable label for particle length internal coordinate
  const VarLabel* d_particle_temperature_label;  ///< Variable label for particle temperature internal coordinate (set equal to d_gas_temperature_label by default if none specified...)
  const VarLabel* d_gas_temperature_label; ///< Variable label for gas temperature (set equal to d_tempINLabel by default if none specified...)

  double d_char_scaling_constant;   ///< Scaling constant for char mass internal coordinate
  double d_length_scaling_constant; ///< Scaling constant for length internal coordinate
  double d_pt_scaling_constant;     ///< Scaling constant for particle temperature internal coordinate

  double d_w_scaling_constant;      ///< Scaling constant for weight
  double d_w_small; // "small" clip value for zero weights

  // ------------------------------
  // Internal coordinate booleans

  bool d_useChar;       ///< Boolean: using char as an internal coordinate?
  bool d_useLength;     ///< Boolean: using length as an internal coordinate?
  bool d_useTparticle;  ///< Boolean: using particle temperature as an internal coordinate?
  bool d_useTgas;       ///< Boolean: using an extra scalar as gas temperature?

  // -----------------------------
  // Other...

  double R_;  ///< [=] kcal/kmol : Ideal gas constant
  double pi_; 

}; // end CharOxidation
} // end namespace Uintah
#endif
