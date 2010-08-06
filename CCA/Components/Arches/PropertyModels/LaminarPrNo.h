#ifndef Uintah_Component_Arches_LaminarPrNo_h
#define Uintah_Component_Arches_LaminarPrNo_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

/** 
* @class  LaminarPrNo
* @author Jeremy T. and Weston E. 
* @date   August 2010
* 
* @brief Computes the laminar Pr number by : ... 
*
* ADD INPUT FILE INFORMATION HERE: 
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label = "label_name" type = "laminar_pr">
*       <temperature>298</temperature>
*     </model>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class LaminarPrNo : public PropertyModelBase {

    public: 

      LaminarPrNo( std::string prop_name, SimulationStateP& shared_state );
      ~LaminarPrNo(); 

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

          LaminarPrNo* build()
          { return scinew LaminarPrNo( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

  }; // class LaminarPrNo
}   // namespace Uintah

#endif
