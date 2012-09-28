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
* See: Hottel & Sarofim, 1978 OR Brad Adam's Thesis (1993)
*
* Here is the UPS spec: 
*
* <model label="my_soot" type="emperical_soot">
   <!-- Emperical soot --> 
   <carbon_content_fuel          spec="REQUIRED DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- carbon content, mass of carbon atoms/mass of fuel --> 
   <carbon_content_ox            spec="REQUIRED DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- carbon content, mass of carbon atoms/mass of oxidizer --> 
   <opl                          spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- this is optional because the radiation model may specify it --> 
   <C1                           spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- default of 0.1, valid values range between 0-0.2 (see Hottel & Sarofim, 1978) --> 
   <E_cr                         spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- Critical equivilence ratio, default 1.0 --> 
   <E_inf                        spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- Equivilence ratio at infinity, default = 2*E_cr --> 
   <E_st                         spec="REQUIRED DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- Equivilence ratio at stoich --> 
   <soot_density                 spec="OPTIONAL DOUBLE 'positive'"
                                 need_applies_to="type empirical_soot"/> <!-- default of 1950.0 --> 
   <density_label                spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/> <!-- default of "density" --> 
   <temperature_label            spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/> <!-- default of "temperature" --> 
   <absorption_label             spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/> <!-- default of "absorpIN" --> 
   <mixture_fraction_label       spec="OPTIONAL STRING"
                                 need_applies_to="type empirical_soot"/> <!-- default of "absorpIN" --> 
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

     double inline get_carbon_content( double f ){ 
       return f * _carb_content_fuel + ( 1.0 - f ) * _carb_content_ox; 
     } 

     double _carb_content_fuel; 
     double _carb_content_ox; 
     double _opl; 
     double _rho_soot; 
     double _C1;                     ///<< Mass fraction of volatile carbon that forms soot, range ( 
     const double _cmw;              ///<< Carbon molecular weight
     double _E_cr;                   ///<< Critical equivilence ratio
     double _E_inf;                  ///<< Infinite equivilence ratio
     double _E_st;                   ///<< Stoichiometric equibilience ratio (mass fuel/mass ox)_st

     const VarLabel* _den_label; 
     const VarLabel* _T_label; 
     const VarLabel* _absorp_label; 
     const VarLabel* _f_label; 

     std::string _den_label_name; 
     std::string _T_label_name; 
     std::string _absorp_label_name; 
     std::string _mf_transform; 
     std::string _f_label_name; 


  }; // class EmpSoot
}   // namespace Uintah

#endif
