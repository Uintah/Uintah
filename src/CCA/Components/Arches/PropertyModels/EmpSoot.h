#ifndef Uintah_Component_Arches_EmpSoot_h
#define Uintah_Component_Arches_EmpSoot_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  EmpSoot
* @author Jeremy Thornock
* @date   5/2012
* 
* @brief Computes the empirical soot volume fraction and the absorption coefficient 
*
* Here is the UPS spec: 
*
* <model label="my_soot" type="emperical_soot">
   <!-- Emperical soot --> 
   <carbon_content               spec="REQUIRED DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> 
   <opl                          spec="REQUIRED DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/>
   <scaling_factor               spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/>
   <c3                           spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/>
   <soot_density                 spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/>
   <density_label                spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/>
   <temperature_label            spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/>
   <absorption_label             spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/>
 </model>
*
*  
*/ 

namespace Uintah{ 

  class EmpSoot : public PropertyModelBase {

    public: 

      EmpSoot( std::string prop_name, SimulationStateP& shared_state );
      ~EmpSoot(); 

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

          EmpSoot* build()
          { return scinew EmpSoot( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

     double _carb_content; 
     double _opl; 
     double _scale_factor;
     double _rho_soot; 
     double _c3; 
     const double _cmw;              //<< Carbon molecular weight

     const VarLabel* _den_label; 
     const VarLabel* _T_label; 
     const VarLabel* _absorp_label; 

     std::string _den_label_name; 
     std::string _T_label_name; 
     std::string _absorp_label_name; 


  }; // class EmpSoot
}   // namespace Uintah

#endif
