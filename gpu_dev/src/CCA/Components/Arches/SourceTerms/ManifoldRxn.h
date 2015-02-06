
#ifndef Uintah_Component_Arches_ManifoldRxn_h
#define Uintah_Component_Arches_ManifoldRxn_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>

namespace Uintah{

class ManifoldRxn: public SourceTermBase {

public: 

  ManifoldRxn( std::string srcName, SimulationStateP& shared_state, 
                std::vector<std::string> reqLabelNames, std::string type );

  ~ManifoldRxn();
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

  /** @brief Schedule initialization */ 
  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, std::vector<std::string> required_label_names, SimulationStateP& shared_state )
        : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){ 
          _type = "constant_src"; 
        };
      ~Builder(){}; 

      ManifoldRxn* build()
      { return scinew ManifoldRxn( _name, _shared_state, _required_label_names, _type ); };

    private: 

      std::string _name; 
      SimulationStateP& _shared_state; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:

  Discretization_new* _disc;   

  std::string _manifold_var_name; 
  std::string _conv_scheme; 

  const VarLabel* _manifold_label; 
  const VarLabel* _old_manifold_label; 
  const VarLabel* _conv_label; 
  const VarLabel* _diff_label; 

  double _prNo; 

}; // end ManifoldRxn
} // end namespace Uintah
#endif
