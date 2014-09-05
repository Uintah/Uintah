#ifndef Uintah_Component_Arches_GlobalCharOxidation_h
#define Uintah_Component_Arches_GlobalCharOxidation_h
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>

//===========================================================================

/**
  * @class    GlobalCharOxidation
  * @author   Charles Reid
  * @date     June 2010
  *
  * @brief    Calculates a global char oxidation reaction rate.
  *
  * @details
  * This class calculates a non-implicit char oxidation reaction rate for a particle j:
  *
  * \f$ r_{hj} = \sum_{l=1}^{N_{\mbox{char rxns}}} \frac{ A_{p} \nu_{s} M_{p} C_{og} }{ \frac{1}{k_r \xi_{p}} + \frac{1}{k_m} } \f$
  *
  * where Ap is the area of the particle, km is the mass transfer coefficient, kr is the rate constant for the
  * heterogeneous char reaction, \f$ \xi_p \f$ is the particle area shape factor, Mp is the MW of reactant (carbon),
  * and nu_s is a stoichiometric ratio.
  *
  * @todo
  * Grab an instance of Properties and/or TabProps to specifically request the
  * species concentrations needed for char oxidation
  *
  * @todo
  * Finish the char reaction rate expression
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ArchesLabel;
class GlobalCharOxidationBuilder: public ModelBuilder 
{
public: 
  GlobalCharOxidationBuilder( const std::string          & modelName,
                              const vector<std::string>  & reqICLabelNames,
                              const vector<std::string>  & reqScalarLabelNames,
                              ArchesLabel          * fieldLabels,
                              SimulationStateP           & sharedState,
                              int qn );

  ~GlobalCharOxidationBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class GlobalCharOxidation: public CharOxidation {
public: 

  GlobalCharOxidation( std::string modelName, 
                       SimulationStateP& shared_state, 
                       ArchesLabel* fieldLabels,
                       vector<std::string> reqICLabelNames, 
                       vector<std::string> reqScalarLabelNames,
                       int qn );

  ~GlobalCharOxidation();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Set the (species) property labels needed by the char oxidation model */
  void setPropertyLabels();

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  ///////////////////////////////////////////////////
  // Get/set methods

  /* getType() defined in CharOxidation parent class */

private:

  double visc;                                   ///< [=] m^2/s : Laminar fluid viscosity
  vector<double>  nu;  ///< Stoiciometric rate constant array (number of Oxygen molecules in oxidizer / number of Oxygen molecules in product) for 4 char reactions: O2, H2, CO2, H2O

}; // end GlobalCharOxidation

} // end namespace Uintah
#endif
