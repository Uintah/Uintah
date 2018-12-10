#ifndef Uintah_Component_Arches_ScalarDissipation_h
#define Uintah_Component_Arches_ScalarDissipation_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>

/** 
 * @class  NormScalarVariance
 * @author Alex Abboud
 * @date   August 2013
 * 
 * @brief Computes the scalar dissipation rate when a
 * "2-equation" scalar variance model is used
 * \f$ \chi = 2 D \tilde{| \nabla Z|^2} \f$ where the term
 * \f$ \tilde{| \nabla Z|^2} \f$ is a transported scalar variable
 */ 

namespace Uintah{ 
  
  class ScalarDissipation : public PropertyModelBase {
    
  public: 
    
    ScalarDissipation( std::string prop_name, MaterialManagerP& materialManager );
    ~ScalarDissipation(); 
    
    void problemSetup( const ProblemSpecP& db ); 
    
    void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
    void computeProp(const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw, 
                     int time_substep );
    
    void sched_initialize( const LevelP& level, SchedulerP& sched );
    void initialize( const ProcessorGroup* pc, 
                    const PatchSubset* patches, 
                    const MaterialSubset* matls, 
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw );
    
    class Builder
    : public PropertyModelBase::Builder { 
      
    public: 
      
      Builder( std::string name, MaterialManagerP& materialManager ) : _name(name), _materialManager(materialManager){};
      ~Builder(){}; 
      
      ScalarDissipation* build()
      { return scinew ScalarDissipation( _name, _materialManager ); };
      
    private: 
      
      std::string _name; 
      MaterialManagerP& _materialManager; 
      
    }; // class Builder 
    
  private: 
    
    std::string _gradmf2_name; 
    
    double _D;                             ///< filtered molecular diffusion coeff.  (assumed constant) 
    
    const VarLabel* _gradmf2_label; 
    
  }; // class ScalarDissipation
}   // namespace Uintah

#endif
