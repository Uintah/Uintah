#ifndef Uintah_Component_Arches_SecondMFMoment_h
#define Uintah_Component_Arches_SecondMFMoment_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

/** 
 * @class  SecondMFMoment
 * @author Alex Abboud
 * @date   August 2013
 * 
 * @brief This calculates the source term to be used in scalar variance
 * transport models for the second moment of the mixture fraction
 * \f$ rate = - \chi * \rho \f$, \f$ \chi \f$ is scalar dissipation
 * and \f$ \rho \f$ is density
 *  
 */ 

namespace Uintah{
  
  class SecondMFMoment: public SourceTermBase {
  public: 
    
    SecondMFMoment( std::string srcName, SimulationStateP& shared_state, 
                                  std::vector<std::string> reqLabelNames, std::string type );
    
    ~SecondMFMoment();
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
      { _type = "squaredmixfrac"; };
      ~Builder(){}; 
      
      SecondMFMoment* build()
      { return scinew SecondMFMoment( _name, _shared_state, _required_label_names, _type ); };
      
    private: 
      
      std::string _name; 
      SimulationStateP& _shared_state; 
      std::vector<std::string> _required_label_names;
      std::string _type;
      
    }; // class Builder 
    
  private:
    
    std::string _density_name;                      ///< String idenifying which expression should be used. 
    std::string _scalarDissipation_name;
    
    const VarLabel* densityLabel;
    const VarLabel* scalarDissLabel;
  }; // end SecondMFMoment
} // end namespace Uintah
#endif
