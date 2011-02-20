#ifndef Uintah_Component_Arches_ABSKP_h
#define Uintah_Component_Arches_ABSKP_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>

/** 
* @class  ABSKP
* @author Julien Pedel
* @date   Feb 2011
* 
* @brief Computes the absorption coefficient from coal particles
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label="abskp" type="absorption_coefficient">
*     </model>
*   </PropertyModels>
* \endcode 
*
* Note that the label here should always be abskp.  If the user uses anything but this name 
* it will be renamed.  
*  
*/ 

namespace Uintah{ 

  class ABSKP : public PropertyModelBase {

    public: 
      typedef std::map< std::string, HeatTransfer*> HeatTransferModelMap;

      ABSKP( std::string prop_name, SimulationStateP& shared_state );
      ~ABSKP(); 

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

          ABSKP* build()
          { return scinew ABSKP( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      HeatTransferModelMap heatmodels_;

  }; // class ABSKP
}   // namespace Uintah

#endif
