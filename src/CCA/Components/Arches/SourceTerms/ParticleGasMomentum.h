#ifndef Uintah_Component_Arches_ParticleGasMomentum_h
#define Uintah_Component_Arches_ParticleGasMomentum_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

/**
  * @class    ParticleGasMomentum
  * @author   Julien Pedel, Charles Reid
  * @date     May 2010
  *           
  * @brief    Creates a source term for momentum coupling between particles and gas.
  *
  */

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

  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      ParticleGasMomentum* build() { 
        return scinew ParticleGasMomentum( _name, _shared_state, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& _shared_state;
      vector<std::string> _required_label_names;
  }; // class Builder

private:


}; // end ParticleGasMomentum
} // end namespace Uintah
#endif
