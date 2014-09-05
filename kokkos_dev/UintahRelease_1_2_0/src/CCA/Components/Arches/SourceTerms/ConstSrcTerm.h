#ifndef Uintah_Component_Arches_ConstSrcTerm_h
#define Uintah_Component_Arches_ConstSrcTerm_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class ConstSrcTermBuilder: public SourceTermBuilder
{
public: 
  ConstSrcTermBuilder(std::string srcName, 
                      vector<std::string> reqLabelNames, 
                      SimulationStateP& sharedState);
  ~ConstSrcTermBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ConstSrcTerm: public SourceTermBase {
public: 

  ConstSrcTerm( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~ConstSrcTerm();
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);
  /** @brief Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  /** @brief Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );
private:

  double d_constant; 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
