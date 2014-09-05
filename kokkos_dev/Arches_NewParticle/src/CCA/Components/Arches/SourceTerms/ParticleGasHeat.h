#ifndef Uintah_Component_Arches_ParticleGasHeat_h
#define Uintah_Component_Arches_ParticleGasHeat_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{ 

//===========================================================================

/**
  * @class    ParticleGasHeat
  * @author   Julien Pedel
  * @date     October 2010
  *           
  * @brief    
  * This class needs documentation.
  *
  */

class ParticleGasHeat: public SourceTermBase {

  public: 

  ParticleGasHeat( std::string src_name, 
                   vector<std::string> required_label_names,
                   SimulationStateP& shared_state );

  ~ParticleGasHeat();

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

  /** @brief  Return a string with the model type */
  inline string getType() {
    return "ParticleGasHeat";
  };

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){}; 
      ParticleGasHeat* build() { 
        return scinew ParticleGasHeat( _name, _required_label_names, _shared_state); 
      }

    private: 

      std::string _name; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 

  }; // class Builder 

private:

  std::string _heat_model_name; 

}; // end ParticleGasHeat
} // end namespace Uintah
#endif
