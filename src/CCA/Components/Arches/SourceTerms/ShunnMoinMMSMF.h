
#ifndef Uintah_Component_Arches_ShunnMoinMMSMF_h
#define Uintah_Component_Arches_ShunnMoinMMSMF_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

class ShunnMoinMMSMF: public SourceTermBase {
public: 

  ShunnMoinMMSMF( std::string srcName, MaterialManagerP& materialManager, 
                std::vector<std::string> reqLabelNames, std::string type );

  ~ShunnMoinMMSMF();
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
          _type = "constant_src"; 
        };
      ~Builder(){}; 

      ShunnMoinMMSMF* build()
      { return scinew ShunnMoinMMSMF( _name, _materialManager, _required_label_names, _type ); };

    private: 

      std::string _name; 
      MaterialManagerP& _materialManager; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:

  double _k; 
  double _w; 
  double _r0, _r1; 
  double _uf, _vf; 
  double _A;

  int _i1; 
  int _i2;

  std::string _plane; 

}; // end ShunnMoinMMSMF
} // end namespace Uintah
#endif
