#ifndef Uintah_Component_Arches_CharOxidationMixtureFraction_h
#define Uintah_Component_Arches_CharOxidationMixtureFraction_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

/**
  * @class    CharOxidationMixtureFraction
  * @author   Charles Reid
  * @date     July 2010
  *           
  * @brief    This is a source term for a coal gas mixture fraction coming from char oxidation.
  *
  */

class CharOxidationMixtureFraction: public SourceTermBase {
public: 

  CharOxidationMixtureFraction( std::string srcName, 
                                SimulationStateP& shared_state, 
                                vector<std::string> reqLabelNames );

  ~CharOxidationMixtureFraction();

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
    return "CharOxidationMixtureFraction";
  };


  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      CharOxidationMixtureFraction* build() { 
        return scinew CharOxidationMixtureFraction( _name, _shared_state, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& _shared_state;
      vector<std::string> _required_label_names;
  }; // class Builder

private:

  string d_charModelName; 

  vector<const VarLabel*> GasModelLabels_;

}; // end CharOxidationMixtureFraction
} // end namespace Uintah
#endif

