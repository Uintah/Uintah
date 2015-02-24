#ifndef Uintah_Component_Arches_NormScalarVariance_h
#define Uintah_Component_Arches_NormScalarVariance_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>


/** 
 * @class  NormScalarVariance
 * @author Alex Abboud
 * @date   August 2013
 * 
 * @brief Computes the normalized scalar variance
 * based on the filtered quantities of mixture fraction and second moment
 * \f$ \tilde{ Z^2 } - \tilde{Z}^2 \f$
 * normalized by the maximum variance based on mixture fraction
 * \f$ \text{Max}_\text{Var} = Z - Z^2 \f$
 * A <Clip> tag exists that will modify the second moment to never be numerically incorrect
 */ 

namespace Uintah{ 
  
  class BoundaryCondition;

  class NormScalarVariance : public PropertyModelBase {
    
  public: 
    
    NormScalarVariance( std::string prop_name, SimulationStateP& shared_state );
    ~NormScalarVariance(); 
    
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
      
      Builder( std::string name, SimulationStateP& shared_state ) : _name(name), _shared_state(shared_state){};
      ~Builder(){}; 
      
      NormScalarVariance* build()
      { return scinew NormScalarVariance( _name, _shared_state ); };
      
    private: 
      
      std::string _name; 
      SimulationStateP& _shared_state; 
    }; // class Builder 
    
  private: 
    
    std::string _mf_label_name; 
    std::string _mf_m2_label_name; 
    
    const VarLabel* _mf_label; 
    const VarLabel* _mf_m2_label; 
    const VarLabel* _vf_label;
    
    bool clip;
    
  protected:
    BoundaryCondition* d_boundaryCondition;
    
    
  }; // class NormScalarVariance
}   // namespace Uintah

#endif
