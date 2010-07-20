#ifndef Uintah_Component_Arches_CharOxidationMixtureFraction_h
#define Uintah_Component_Arches_CharOxidationMixtureFraction_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

/**
  * @class    CharOxidationMixtureFraction
  * @author   Charles Reid
  * @date     July 2010
  *           
  * @brief    This is a source term for a coal gas mixture fraction coming from char oxidation.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class CharOxidationMixtureFractionBuilder: public SourceTermBuilder
{
public: 
  CharOxidationMixtureFractionBuilder(std::string srcName, 
                                      vector<std::string> reqLabelNames, 
                                      SimulationStateP& sharedState);
  ~CharOxidationMixtureFractionBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class CharOxidationMixtureFraction: public SourceTermBase {
public: 

  CharOxidationMixtureFraction( std::string srcName, 
                                SimulationStateP& shared_state, 
                                vector<std::string> reqLabelNames );

  ~CharOxidationMixtureFraction();

  /** @brief  Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief  Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief  Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief  Return a string with the model type */
  string getType() {
    return "CharOxidationMixtureFraction";
  };

private:

  string d_charModelName; 

  vector<const VarLabel*> GasModelLabels_;

}; // end CharOxidationMixtureFraction
} // end namespace Uintah
#endif

