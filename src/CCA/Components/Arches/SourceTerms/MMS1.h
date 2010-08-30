#ifndef Uintah_Component_Arches_MMS1_h
#define Uintah_Component_Arches_MMS1_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

/**
  * @class    MMS1
  * @author   Jeremy Thornock
  * @date     
  *           
  * @brief    
  * This source term is for an MMS verification.
  * The MMS function is:
  * \f[ 
    RHS = 2*\pi*cos(2*\pi*x)*cos(2*\pi*y) - 2*\pi*sin(2*\pi*x)*sin(2*\pi*y)
  * \f]
  * This MMS can be used by pretty much any scalar equation.
  *
  */

class MMS1: public SourceTermBase {
public: 

  MMS1( std::string srcName, SimulationStateP& shared_state, 
        vector<std::string> reqLabelNames );

  ~MMS1();

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
    return "MMS1";
  };

  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      MMS1* build() { 
        return scinew MMS1( _name, _shared_state, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& _shared_state;
      vector<std::string> _required_label_names;
  }; // class Builder

private:

}; // end MMS1
} // end namespace Uintah
#endif
