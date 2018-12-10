#ifndef Uintah_Component_Arches_CoalGasDevol_h
#define Uintah_Component_Arches_CoalGasDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{ 

class CoalGasDevol: public SourceTermBase {

  public: 

  CoalGasDevol( std::string src_name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager, std::string type );

  ~CoalGasDevol();

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
          _type = "coal_gas_devol"; 
        };
      ~Builder(){}; 

      CoalGasDevol* build()
      { return scinew CoalGasDevol( _name, _required_label_names, _materialManager, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      MaterialManagerP& _materialManager; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:
  std::string m_rcmass_root;
  std::string _devol_model_name; 
  const VarLabel* m_tar_src_label;
  std::string m_tar_src_name;        ///< string name for tar src 
  const VarLabel* m_devol_for_nox_src_label;
  std::string m_devol_for_nox_src_name;        ///< string name for devol src used for nox 
  const VarLabel* m_devol_bd_src_label;
  std::string m_devol_bd_src_name;        ///< string name for bd src used in the nox model
  double m_tarFrac;
  double m_lightFrac;
  double m_v_hiT;
  bool m_dest_flag;// flag indicating whether or not deposition mass will be added to the gas phase. 

}; // end CoalGasDevol
} // end namespace Uintah
#endif
