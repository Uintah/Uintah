#ifndef Uintah_Component_Arches_LinearSwelling_h
#define Uintah_Component_Arches_LinearSwelling_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

//===========================================================================

/**
  * @class    LinearSwelling
  * @author   Milo Parra, Ben Isaac
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class LinearSwellingBuilder: public ModelBuilder 
{
public: 
  LinearSwellingBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                SimulationStateP                & sharedState,
                                int qn );

  ~LinearSwellingBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class LinearSwelling: public ModelBase {
public: 

  LinearSwelling( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~LinearSwelling();
  
  typedef std::map< std::string, ModelBase*> ModelMap;
  typedef std::map< std::string, Devolatilization*> DevolModelMap;

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of some special/local variables */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local variables */
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
                     const int timeSubStep );
  
  inline std::string getType() {
    return "Constant"; }

private:

  const VarLabel* m_devolRCLabel;
  const VarLabel* m_weighted_length_label;
  const VarLabel* m_birth_label;
  const VarLabel* m_scaled_weight_varlabel;
  const VarLabel* m_RHS_source_varlabel;
  const VarLabel* m_RHS_weight_varlabel;
  
  double m_init_diam;   ///< initial particle size 
  double m_Fsw;   ///< sweling factor default is 1.05 
  double m_v_hiT;   ///< ultimate yield from FOWY 
  double m_init_rc;   ///< initial raw coal mass 
  double m_scaling_const_length;   ///< scaling constant for length transport 
  double m_scaling_const_rc;   ///< scaling constant for raw coal transport 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
