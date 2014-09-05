#ifndef Uintah_Component_Arches_DevolMixtureFraction_h
#define Uintah_Component_Arches_DevolMixtureFraction_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

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

class DevolMixtureFraction: public SourceTermBase {

  public: 

  DevolMixtureFraction( std::string srcName, 
                        SimulationStateP& shared_state, 
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
  
  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      DevolMixtureFraction* build() { 
        return scinew DevolMixtureFraction( _name, _shared_state, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& _shared_state;
      vector<std::string> _required_label_names;
  }; // class Builder

private:

  string d_devolModelName; 

}; // end DevolMixtureFraction
} // end namespace Uintah
#endif

