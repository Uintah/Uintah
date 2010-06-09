#ifndef Uintah_Component_Arches_MMS1_h
#define Uintah_Component_Arches_MMS1_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class MMS1Builder: public SourceTermBuilder
{
public: 
  MMS1Builder(std::string srcName, 
              vector<std::string> reqLabelNames, 
              SimulationStateP& sharedState);
  ~MMS1Builder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class MMS1: public SourceTermBase {
public: 

  MMS1( std::string srcName, SimulationStateP& shared_state, 
        vector<std::string> reqLabelNames );

  ~MMS1();

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
    return "MMS1";
  };

private:

}; // end MMS1
} // end namespace Uintah
#endif
