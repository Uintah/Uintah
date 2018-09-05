#ifndef Uintah_Component_Arches_ShrinkageRate_h
#define Uintah_Component_Arches_ShrinkageRate_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    ShrinkageRate
  * @author   Milo Parra, Ben Isaac
  * @date     March 2017        Check-in of initial version
  *           March 2017   Verification
  *
  * @brief    Model for the shrinkage of coal particles, based on Char oxidation
  *
  * The Builder is required because of the Model Factory; the Factory needs
  * some way to create the model term and register it.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ShrinkageRateBuilder: public ModelBuilder 
{
public: 
  ShrinkageRateBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                MaterialManagerP                & materialManager,
                                int qn );

  ~ShrinkageRateBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ShrinkageRate: public ModelBase {
public: 

  ShrinkageRate( std::string modelName, 
                         MaterialManagerP& materialManager, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~ShrinkageRate();

  ////////////////////////////////////////////////
  // Initialization method
  typedef std::map< std::string, CharOxidation*> CharOxiModelMap;

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

  /** @brief  */
    
  const VarLabel* m_weight_scaled_varlabel; // label for scaled weights
  const VarLabel* m_charoxiSize_varlabel; // label for particle shrinkage rate

  
  double m_scaling_const_length;   ///< Scaling constant for the particle size
  
}; // end ConstSrcTerm
} // end namespace Uintah
#endif
