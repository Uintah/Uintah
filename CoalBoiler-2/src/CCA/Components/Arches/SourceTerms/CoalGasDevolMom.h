#ifndef Uintah_Component_Arches_CoalGasDevolMom_h
#define Uintah_Component_Arches_CoalGasDevolMom_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{ 

class CoalGasDevolMom: public SourceTermBase {

  public: 

  CoalGasDevolMom( std::string src_name, std::vector<std::string> required_label_names, ArchesLabel* field_labels, SimulationStateP& shared_state, std::string type );

  ~CoalGasDevolMom();

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

      Builder( std::string name, std::vector<std::string> required_label_names, ArchesLabel* field_labels, SimulationStateP& shared_state )
        : _name(name), _required_label_names(required_label_names), _field_labels(field_labels), _shared_state(shared_state){ 
          _type = "coal_gas_devol_mom"; 
        };
      ~Builder(){}; 

      CoalGasDevolMom* build()
      { return scinew CoalGasDevolMom( _name, _required_label_names, _field_labels,  _shared_state, _type ); };

    private: 

      std::string _name; 
      std::vector<std::string> _required_label_names;
      ArchesLabel* _field_labels; 
      SimulationStateP& _shared_state; 
      std::string _type; 

  }; // class Builder 

private:

  std::string _devol_model_name; 
  const VarLabel* _u_varlabel;
  ArchesLabel* _field_labels; 

}; // end CoalGasDevolMom
} // end namespace Uintah
#endif
