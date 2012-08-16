#ifndef Uintah_Component_Arches_ScalarDiss_h
#define Uintah_Component_Arches_ScalarDiss_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

/** 
* @class  ScalarDiss
* @author Jeremy Thornock
* @date   Aug 2011
* 
* @brief Computes the scalar turbulent dissipation rate 
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label="scalar_dissipation_rate type="scalar_diss">
*       <constant_diss/>                  <!-- tells the code to use a constant value for scalar dissipation-->
*       <constant>DOUBLE</constant>       <!-- if using constant dissipation, use this value --> 
*     </model>
*   </PropertyModels>
* \endcode 
*
* Note that the label here should always be scalar_dissipation_rate.  If the user uses anything but this name 
* it will be renamed.  
*  
*/ 

namespace Uintah{ 

  class ScalarDiss : public PropertyModelBase {

    public: 

      ScalarDiss( std::string prop_name, SimulationStateP& shared_state );
      ~ScalarDiss(); 

      void problemSetup( const ProblemSpecP& db ); 

      void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
      void computeProp(const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, 
                       int time_substep );

      void sched_dummyInit( const LevelP& level, SchedulerP& sched );
      void dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw );

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

          ScalarDiss* build()
          { return scinew ScalarDiss( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      double _constant_diss;                              ///< Constant scalar dissipation rate

  }; // class ScalarDiss
}   // namespace Uintah

#endif
