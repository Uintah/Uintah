#ifndef Uintah_Component_Arches_PCTransport_h
#define Uintah_Component_Arches_PCTransport_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

// SEE SOURCETEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  ADD
* @author ADD
* @date   ADD
* 
* @brief Computes ADD INFORMATION HERE
*
* ADD INPUT FILE INFORMATION HERE: 
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <Sources>
*     <src label="STRING" type="?????">
*         .....ADD DETAILS....
*     </src>
*   </Sources>
* \endcode 
*  
*/ 

namespace Uintah{

class PCTransport: public SourceTermBase {
public: 

  typedef std::map<int, constCCVariable<double> > PcStorage;         

  PCTransport( std::string srcName, SimulationStateP& shared_state, 
                std::vector<std::string> reqLabelNames, std::string type );
  ~PCTransport();

  void problemSetup(const ProblemSpecP& db);
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );
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
          _type="pctransport";
        };
      ~Builder(){}; 

      PCTransport* build()
      { return scinew PCTransport( _name, _shared_state, _required_label_names, _type ); };

    private: 

      std::string _name; 
      SimulationStateP& _shared_state; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:

  std::string _pc_scal_file; 
  std::string _pc_st_scal_file; 
  std::string _svm_base_name;
  std::vector<std::string> _svm_models;
  std::map<int, std::string> _pc_info; 
  std::map<int, const VarLabel*> _pc_labels; 

  int _N_PCS;
  int _N_STS; 
  int _N_IND; 
  int _N_TOT;

  /** @brief Looks for all the PC labels and makes sure they exist **/
  void check_for_pc_labels(); 

}; // end PCTransport
} // end namespace Uintah
#endif
