#ifndef Uintah_Component_Arches_ParticleGasMomentum_h
#define Uintah_Component_Arches_ParticleGasMomentum_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class ParticleGasMomentumBuilder: public SourceTermBuilder
{
public: 
  ParticleGasMomentumBuilder(std::string srcName, 
                      vector<std::string> reqLabelNames, 
                      SimulationStateP& sharedState);
  ~ParticleGasMomentumBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ParticleGasMomentum: public SourceTermBase {
public: 

  ParticleGasMomentum( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~ParticleGasMomentum();

  /** @brief  Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Schedule: Compute the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief  Compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief  Schedule: Dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  
  /** @brief  Dummy initialization */ 
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief  Return a string with the model type */
  string getType() {
    return "ParticleGasMomentum";
  };

private:

  //double d_constant; 
  //std::string d_dragModelName;

}; // end ParticleGasMomentum
} // end namespace Uintah
#endif
