#ifndef Uintah_Component_Arches_fvSootFromYsoot_h
#define Uintah_Component_Arches_fvSootFromYsoot_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  fvSootFromYsoot
* @author David Lignell
* @date   4/2015
* 
* @brief Computes the soot volume fraction from the soot mass fraction (which is assumed transported)
*
* Here is the UPS spec: 
*
*
*  
*/ 

namespace Uintah{ 

  class fvSootFromYsoot : public PropertyModelBase {

    public: 

      fvSootFromYsoot( std::string prop_name, SimulationStateP& shared_state );
      ~fvSootFromYsoot(); 

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

          fvSootFromYsoot* build()
          { return new fvSootFromYsoot( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

     const VarLabel* _den_label; 
     const VarLabel* _Ys_label; 
     const VarLabel* _absorp_label;  

     std::string _den_label_name; 
     std::string _Ys_label_name; 
     std::string _absorp_label_name;

  }; // class fvSootFromYsoot
}   // namespace Uintah

#endif
