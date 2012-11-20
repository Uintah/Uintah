#ifndef Uintah_Component_Arches_ScalarVarianceScaleSim_h
#define Uintah_Component_Arches_ScalarVarianceScaleSim_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  ADD
* @author ADD
* @date   ADD
* 
* @brief Computes ADD INFORMATION HERE
*
* ADD INPUT FILE INFORMATION HERE: 
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <.......>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class Filter; 

  class ScalarVarianceScaleSim : public PropertyModelBase {

    public: 

      ScalarVarianceScaleSim( std::string prop_name, SimulationStateP& shared_state );
      ~ScalarVarianceScaleSim(); 

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

          ScalarVarianceScaleSim* build()
          { return scinew ScalarVarianceScaleSim( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

          Filter* _filter; 

      }; // class Builder 

    private: 

      std::string _mf_label_name; 
      std::string _density_label_name; 

      const VarLabel* _mf_label; 
      const VarLabel* _density_label; 

      Filter* _filter; 

  }; // class ScalarVarianceScaleSim
}   // namespace Uintah

#endif
