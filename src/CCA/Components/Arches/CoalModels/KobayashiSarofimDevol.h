#ifndef Uintah_Component_Arches_KobayashiSarofimDevol_h
#define Uintah_Component_Arches_KobayashiSarofimDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah{

//===========================================================================

/**
  * @class    KobayashiSarofimDevol
  * @author   Jeremy Thornock, Julien Pedel, Charles Reid
  * @date     May 2009      : Check-in of initial version \n
  *           November 2009 : Verification \n
  *           July 2010     : Cleanup \n
  *
  * @brief    A class for calculating the raw coal internal coordinate
  *           model term for the Kobayashi-Sarofim coal devolatilization model.
  *
  * @todo     
  * Add calculation of char production model term. (But first, the model list storage in DQMOMEqn class must be fixed)
  *
  */

//---------------------------------------------------------------------------
// Builder
class ArchesLabel;
class KobayashiSarofimDevolBuilder: public ModelBuilder 
{
public: 
  KobayashiSarofimDevolBuilder( const std::string          & modelName,
                                const vector<std::string>  & reqICLabelNames,
                                const vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel          * fieldLabels,
                                SimulationStateP           & sharedState,
                                int qn );

  ~KobayashiSarofimDevolBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class KobayashiSarofimDevol: public Devolatilization {
public: 

  KobayashiSarofimDevol( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  ~KobayashiSarofimDevol();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the dummy initialization required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

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

  // TODO: add Glacier computation methods

  /** @brief  Get raw coal reaction rate (see Glacier) */
  double calcRawCoalReactionRate() {
    return 0; }

  /** @brief  Get gas volatile production rate (see Glacier) */
  double calcGasDevolRate() {
    return 0; }

  /** @brief  Get char production rate (see Glacier) */
  double calcCharProductionRate() {
    return 0; }

  ///////////////////////////////////////////////////
  // Get/set methods

  /* getType method is defined in parent class... */

private:

  const VarLabel* d_charModelLabel; ///< VarLabel for model term G for char creation

  double A1;        ///< Pre-exponential factors for devolatilization rate constants
  double A2;        ///< Pre-exponential factors for devolatilization rate constants
  double E1;        ///< Activation energy for devolatilization rate constant
  double E2;        ///< Activation energy for devolatilization rate constant
  double Y1_;       ///< Volatile fraction from proximate analysis
  double Y2_;       ///< Fraction devolatilized at higher temperatures (often near unity)
  double k1;        ///< Rate constant for devolatilization reaction 1
  double k2;        ///< Rate constant for devolatilization reaction 2
  
  double R_;        ///< Ideal gas constant
  
  bool d_useRawCoal;   ///< Boolean: is a DQMOM internal coordinate specified for raw coal mass?
  bool d_useChar;      ///< Boolean: is a DQMOM internal coordinate specified for char mass?
  bool d_useTparticle; ///< Boolean: is a DQMOM internal coordinate specified for particle temperature?
  bool d_useTgas;      ///< Boolean: is a scalar variable specified for gas temperature?

}; // end KobyaashiSarofimDevol
} // end namespace Uintah
#endif
