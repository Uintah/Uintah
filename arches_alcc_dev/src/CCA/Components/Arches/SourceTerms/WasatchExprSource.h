#ifndef Uintah_Component_Arches_WasatchExprSource_h
#define Uintah_Component_Arches_WasatchExprSource_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  
  class WasatchExprSource: public SourceTermBase {
  public: 
    
    WasatchExprSource( std::string srcName, SimulationStateP& shared_state, 
                      vector<std::string> reqLabelNames, std::string type );
    
    ~WasatchExprSource();
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
      : _name(name), 
      _shared_state(shared_state), 
      _required_label_names(required_label_names)
      { _type = "wasatchexpr"; };
      ~Builder(){}; 
      
      WasatchExprSource* build()
      { return scinew WasatchExprSource( _name, _shared_state, _required_label_names, _type ); };
      
    private: 
      
      std::string _name; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 
      std::string _type;
      
    }; // class Builder 
    
  private:
    
    std::string _was_expr;                      ///< String idenifying which expression should be used. 
    
  }; // end WasatchExprSource
} // end namespace Uintah
#endif
