#ifndef Uintah_Component_Arches_TabRxnRate_h
#define Uintah_Component_Arches_TabRxnRate_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

class TabRxnRate: public SourceTermBase {
public: 

  TabRxnRate( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames, std::string type );

  ~TabRxnRate();
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

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, vector<std::string> required_label_names, SimulationStateP& shared_state ) 
        : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){
          _type = "tab_rxn_rate"; 
        };
      ~Builder(){}; 

      TabRxnRate* build()
      { return scinew TabRxnRate( _name, _shared_state, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 

  }; // class Builder 

private:

  std::string _rxn_rate;                      ///< String idenifying which rate should be used. 

}; // end TabRxnRate
} // end namespace Uintah
#endif
