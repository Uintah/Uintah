#ifndef Uintah_Component_Arches_MMS_Z_h
#define Uintah_Component_Arches_MMS_Z_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

namespace Uintah{

//===========================================================================

/**
  * @class    MMS_Z
  * @author   Charles Reid
  * @date     May 19, 2011
  *           
  * @brief    
  * This class creates a source term intended for MMS verification of
  * the Y convection term only.
  */

class MMS_Z: public SourceTermBase {
public: 

  MMS_Z( std::string srcName, SimulationStateP& shared_state, 
        vector<std::string> reqLabelNames );

  ~MMS_Z();

  /** @brief  Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief  Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief  Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  static double evaluate_MMS( double x,
                              double y,
                              double z,
                              Vector domain_size ) {
    double function;
    double pi = MMS_Z::pi;
    function =  cos( 4.0*pi*(y/domain_size.y()) );
    function *= sin( 4.0*pi*(z/domain_size.z()) );
    return function;
  }

  static double evaluate_MMS_source( double x,
                                     double y,
                                     double z,
                                     Vector domain_size ) {
    double source;
    double pi = MMS_Z::pi;
    source =  4.0*pi*MMS_Z::d_wvel*(1/domain_size.z());
    source *= cos( 4.0*pi*(y/domain_size.y()) );
    source *= cos( 4.0*pi*(z/domain_size.z()) );
    return source;
  }

  /** @brief  Return a string with the model type */
  string getType() {
    return "MMS_Z";
  };

  class Builder : public SourceTermBase::Builder {
    public:
      Builder( std::string name, 
               vector<std::string> required_label_names, 
               SimulationStateP& shared_state ) : 
               _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){};
      MMS_Z* build() { 
        return scinew MMS_Z( _name, _shared_state, _required_label_names ); 
      }
    private:
      std::string _name;
      SimulationStateP& _shared_state;
      vector<std::string> _required_label_names;
  }; // class Builder

private:

  GridP grid;

  static double d_uvel; ///< U velocity (specified by Arches constant MMS)
  static double d_vvel; ///< V velcoity (specified by Arches constant MMS)
  static double d_wvel; ///< W velcoity (specified by Arches constant MMS)

  static double pi;

}; // end MMS_Z
} // end namespace Uintah

#endif

