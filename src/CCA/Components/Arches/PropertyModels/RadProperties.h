#ifndef Uintah_Component_Arches_RadProperties_h
#define Uintah_Component_Arches_RadProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/ArchesLabel.h>

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

  class BoundaryCondition_new;
  class RadPropertyCalculator; 

  class RadProperties : public PropertyModelBase {

    public: 

      RadProperties( std::string prop_name, SimulationStateP& shared_state, ArchesLabel  * d_fieldLabel );
      ~RadProperties(); 



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

          Builder( std::string name, SimulationStateP& shared_state,ArchesLabel * fieldLabels) : _name(name), _shared_state(shared_state),_fieldLabels(fieldLabels) {};
          ~Builder(){}; 

          RadProperties* build()
          { return new RadProperties( _name, _shared_state, _fieldLabels ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 
          ArchesLabel* _fieldLabels;

      }; // class Builder 

    private: 

      int _nQn_part ;                                // number of quadrature nodes in DQMOM
      std::string  _base_temperature_label_name;          // DQMOM Temperature name
      std::string  _base_size_label_name;                 // DQMOM size_name

      ArchesLabel * _fieldLabels;
      bool  _particlesOn ;
      bool  _scatteringOn ;
      RadPropertyCalculator::PropertyCalculatorBase* _calc; 
      RadPropertyCalculator::opticalPropertyCalculatorBase* _ocalc; // needed when including scattering
      const VarLabel* _temperature_label; 
      std::string _temperature_name;
      BoundaryCondition_new* _boundaryCond;

  }; // class RadProperties
}   // namespace Uintah

#endif
