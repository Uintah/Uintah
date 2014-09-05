#ifndef Uintah_Component_Arches_DissipationSource_h
#define Uintah_Component_Arches_DissipationSource_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

/** 
 * @class  DissipationSource
 * @author Alex Abboud
 * @date   August 2013
 * 
 * @brief This calculates the source term to be used for the 
 * term in the 2-equation scalar variance model for dissipation
 * the acutal dissipation is \f$ \chi = 2 D \tilde{| \nabla Z |^2} \f$, this source term is
 * to be used for the transport equation of \f$ \tilde{| \nabla Z | ^2} \f$
 * the source term is 4 parts resolved & unresolved production & dissipation
 * \f$ = - 2 \rho \left( \frac{\partial u_i}{\partial x_j}\frac{\partial Z}{\parital x_i}\frac{\partial Z}{\parital x_j} \right)
 *       + C_prd \rho \frac{32 \mu_t}{\Delta^2} ( \tilde{| \nabla |^2} - | \nabla \tilde{Z} |^2 )
 *       - 2 \rho D \left( \frac{\partial^2 Z}{\partial x_i \partial x_j} \right)
 *       - \frac{12 D}{\Delta^2 | \nabla Z |^2} ( \tilde{| \nabla |^2} - | \nabla \tilde{Z} |^2 )^2 \f$
 *
 */ 

namespace Uintah{
  
  class DissipationSource: public SourceTermBase {
  public: 
    
    DissipationSource( std::string srcName, SimulationStateP& shared_state, 
                       std::vector<std::string> reqLabelNames, std::string type );
    
    ~DissipationSource();
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
      
      Builder( std::string name, std::vector<std::string> required_label_names, SimulationStateP& shared_state )
      : _name(name), 
      _shared_state(shared_state), 
      _required_label_names(required_label_names)
      { _type = "dissipation_src"; };
      ~Builder(){}; 
      
      DissipationSource* build()
      { return scinew DissipationSource( _name, _shared_state, _required_label_names, _type ); };
      
    private: 
      
      std::string _name; 
      SimulationStateP& _shared_state; 
      std::vector<std::string> _required_label_names;
      std::string _type;
      
    }; // class Builder 
    
  private:
    
    double _D; //diffusive coefficient
    
    std::string _density;                      ///< String idenifying which expression should be used. 
    std::string _mixfrac;
    std::string _grad_mixfrac2;
    std::string _ccvel;
    std::string _mu_t;
    
    const VarLabel* _densityLabel;
    const VarLabel* _mixFracLabel;
    const VarLabel* _gradMixFrac2Label;
    const VarLabel* _ccVelocityLabel;
    const VarLabel* _turbViscLabel;
    const VarLabel* _volfrac_label;
    
  }; // end DissipationSource
} // end namespace Uintah
#endif
