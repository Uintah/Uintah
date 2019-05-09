#ifndef Uintah_Component_Arches_CoalGasOxi_h
#define Uintah_Component_Arches_CoalGasOxi_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{ 

class CoalGasOxi: public SourceTermBase {

  public: 

  CoalGasOxi( std::string src_name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager, std::string type );

  ~CoalGasOxi();

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

      Builder( std::string name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager )
        : _name(name), _materialManager(materialManager), _required_label_names(required_label_names){
          _type = "coal_gas_oxi"; 
        };
      ~Builder(){}; 

      CoalGasOxi* build()
      { return scinew CoalGasOxi( _name, _required_label_names, _materialManager, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      MaterialManagerP& _materialManager; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:
  std::string m_charmass_root;
  std::string _oxi_model_name; 
  bool m_dest_flag;// flag indicating whether or not deposition mass will be added to the gas phase. 
  const VarLabel* m_char_for_nox_src_label;
  std::string m_char_for_nox_src_name;        ///< string name for char src used for nox 
  const VarLabel* m_char_bd_src_label;
  std::string m_char_bd_src_name;        ///< string name for bd src used in the nox model

}; // end CoalGasOxi
} // end namespace Uintah
#endif
