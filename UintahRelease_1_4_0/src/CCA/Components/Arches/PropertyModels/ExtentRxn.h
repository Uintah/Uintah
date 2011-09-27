#ifndef Uintah_Component_Arches_ExtentRxn_h
#define Uintah_Component_Arches_ExtentRxn_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  ExtentRxn
* @author Jeremy Thornock
* @date   Aug 2011
* 
* @brief Computes the extent of reaction for any grid transported variable
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label=STRING type="extent_rxn">
*       <fuel_mass_fraction>DOUBLE</fuel_mass_fraction>       <!-- mass fraction of species in the fuel --> 
*       <scalar_label>STRING</scalar_label>                   <!-- name of the scalar --> 
*       <mix_frac_label>STRING</mix_frac_label>               <!-- name of the mixture fraction --> 
*     </model> 
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class ExtentRxn : public PropertyModelBase {

    public: 

      ExtentRxn( std::string prop_name, SimulationStateP& shared_state );
      ~ExtentRxn(); 

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

          ExtentRxn* build()
          { return scinew ExtentRxn( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      const VarLabel* _strip_label; 

      std::string _scalar_name; 
      std::string _mixture_fraction_name; 

      double _fuel_mass_frac; 

  }; // class ExtentRxn
}   // namespace Uintah

#endif
