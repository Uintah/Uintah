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
* @date   9/2015
* 
* @brief Computes the soot volume fraction from the soot mass fraction (which is assumed transported) and the absorption coefficient
*
* Here is the UPS spec: 
*
* <model label="my_soot" type="fv_soot">
   <!-- fv_soot --> 
   <opl                          spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type fv_soot"/> <!-- this is optional because the radiation model may specify it --> 
   <soot_density                 spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type fv_soot"/> <!-- default of 1950.0 --> 
   <density_label                spec="OPTIONAL STRING"
                                 need_applies_to="type fv_soot"/> <!-- default of "density" --> 
   <temperature_label            spec="OPTIONAL STRING"
                                 need_applies_to="type fv_soot"/> <!-- default of "temperature" --> 
   <absorption_label             spec="OPTIONAL STRING"
                                 need_applies_to="type fv_soot"/> <!-- default of "absorpIN" -->
   <Ysoot_label			 spec="OPTIONAL STRING"
				 need_applies_to="type fv_soot"/> 
 </model>
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
          { return scinew fvSootFromYsoot( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

     double _rho_soot;
     double _opl;

     const VarLabel* _den_label; 
     const VarLabel* _T_label;
     const VarLabel* _Ys_label; 
     const VarLabel* _absorp_label;  

     std::string _den_label_name;
     std::string _T_label_name; 
     std::string _Ys_label_name; 
     std::string _absorp_label_name;

  }; // class fvSootFromYsoot
}   // namespace Uintah

#endif
