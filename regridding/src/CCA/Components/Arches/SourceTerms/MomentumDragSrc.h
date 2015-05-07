#ifndef Uintah_Component_Arches_MomentumDragSrc_h
#define Uintah_Component_Arches_MomentumDragSrc_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
/**
 * @class  MomentumDragSrc
 * @author Alex Abboud
 * @date   Sept 2014
 *
 * @brief Assembles source term for the particle drag from the
 *        particle phase.  This is generalized to be used for testing in
 *        1/2D examples and should be able to be used for any particle method
 *        as the labels are inputs rather than fixed 
 *
 *
 *
 * Input file interface is as follows:
 \code
 <Sources>
 <src label="STRING REQUIRED" type="momentum_drag_src"/>
 </Sources>
 \endcode
 *
 */

namespace Uintah{
  
  class MomentumDragSrc: public SourceTermBase {
  public:
    
    MomentumDragSrc( std::string srcName, SimulationStateP& shared_state,
                     std::vector<std::string> reqLabelNames, std::string type );
    
    ~MomentumDragSrc();
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
      : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){
        _type = "momentum_drag_src";
      };
      ~Builder(){};
      
      MomentumDragSrc* build()
      { return scinew MomentumDragSrc( _name, _shared_state, _required_label_names, _type ); };
      
    private:
      
      std::string _name;
      std::string _type;
      SimulationStateP& _shared_state;
      std::vector<std::string> _required_label_names;
      
    }; // Builder
  private:
    
    //double d_constant;
    std::string d_dragModelName;
    
    std::string _base_x_drag;
    std::string _base_y_drag;
    std::string _base_z_drag;
    
    int _N;
    
  }; // end MomentumDragSrc
} // end namespace Uintah
#endif
