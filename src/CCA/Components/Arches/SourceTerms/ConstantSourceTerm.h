#ifndef Uintah_Component_Arches_ConstantSourceTerm_h
#define Uintah_Component_Arches_ConstantSourceTerm_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

/**
  * @class    Constant Source Term
  * @author   ?
  * @date     
  *           
  * @brief    
  * This is a constant source term.  
  * It make the RHS of a scalar equation equal to a constant.
  *
  */

class ConstantSourceTerm: public SourceTermBase {
public: 

  ConstantSourceTerm( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~ConstantSourceTerm();

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
    return "ConstantSourceTerm";
  };

  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), d_sharedState(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      ConstantSourceTerm* build() { 
        return scinew ConstantSourceTerm( _name, d_sharedState, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& d_sharedState;
      vector<std::string> _required_label_names;
  }; // class Builder

private:

  double d_constant; 

}; // end ConstantSourceTerm
} // end namespace Uintah
#endif
