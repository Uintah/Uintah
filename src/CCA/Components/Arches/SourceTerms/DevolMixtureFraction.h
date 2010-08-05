#ifndef Uintah_Component_Arches_DevolMixtureFraction_h
#define Uintah_Component_Arches_DevolMixtureFraction_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

/**
  * @class    DevolMixtureFraction
  * @author   Jeremy Thornock
  * @date     
  *           
  * @brief    
  * This is a source term for a coal gas mixture fraction
  * coming from devolatilization.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class DevolMixtureFractionBuilder: public SourceTermBuilder
{
public: 
  DevolMixtureFractionBuilder(std::string srcName, 
                              vector<std::string> reqLabelNames, 
                              SimulationStateP& sharedState);
  ~DevolMixtureFractionBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class DevolMixtureFraction: public SourceTermBase {
public: 

  DevolMixtureFraction( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~DevolMixtureFraction();

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
  inline string getType() {
    return "DevolMixtureFraction";
  };

private:

  string d_devolModelName; 

}; // end DevolMixtureFraction
} // end namespace Uintah
#endif

