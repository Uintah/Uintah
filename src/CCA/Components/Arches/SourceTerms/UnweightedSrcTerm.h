#ifndef Uintah_Component_Arches_UnweightedSrcTerm_h
#define Uintah_Component_Arches_UnweightedSrcTerm_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

class ArchesLabel;
class UnweightedSrcTerm: public SourceTermBase {
public: 

  UnweightedSrcTerm( std::string srcName, 
                     SimulationStateP& shared_state, 
                     vector<std::string> reqLabelNames );

  UnweightedSrcTerm( std::string srcName,
                     SimulationStateP& shared_state,
                     vector<string> reqLabelNames,
                     ArchesLabel* fieldLabels );

  ~UnweightedSrcTerm();

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
  
  inline string getType() {
    return "UnweightedSrcTerm";
  };

  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state,
               ArchesLabel* fieldLabels ) : 
               _name(name), d_sharedState(shared_state), _required_label_names(required_label_names), d_fieldLabels(fieldLabels) {};
      ~Builder(){};
      UnweightedSrcTerm* build() { 
        return scinew UnweightedSrcTerm( _name, d_sharedState, _required_label_names, d_fieldLabels ); 
      }
    private:
      std::string _name;
      SimulationStateP& d_sharedState;
      vector<std::string> _required_label_names;
      ArchesLabel* d_fieldLabels;
  }; // class Builder

private:

  double d_constant; 

  const VarLabel* d_particle_velocity_label;

  ArchesLabel* d_fieldLabels;

}; // end UnweightedSrcTerm
} // end namespace Uintah
#endif
