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

      // extra local labels:
      const VarLabel* _mu_label;            ///< viscosity label 
      const VarLabel* _D_label;             ///< diffusion coefficient label

      // input information
      std::string _mix_frac_label_name;     ///< label name for the mixture fraction used for this model 
      double      _pressure;                ///< atmospheric pressure
      bool        _binary_mixture;          ///< if binary mixture 
      // -- for binary mixtures: 
      // mixture faction is defined as: 
      // f = [kg of a] / [kg of a + kg of b]
      double      _molar_mass_a;              ///< molar mass for species a (fuel)  
      double      _molar_mass_b;              ///< molar mass for species b (air)
      double      _norm_boil_pt_a;            ///< normal boiling point species a
      double      _norm_boil_pt_b;            ///< normal boiling point species b
      

      //  -------------------->>> Inline property evaluators <<<--------------------- 
      inline double getVisc( double f, double T ){

        //  compute viscosity here 
        if ( _binary_mixture ) {

        } else {

          throw InvalidValue("Error: For laminar Pr number property, only binary mixtures are currently supported", __FILE__, __LINE__); 
        } 

      }; 

      inline double getDiffCoef( double f, double T ){

        // compute diffusion coeffficient here
        if ( _binary_mixture ) {

        } else {

          throw InvalidValue("Error: For laminar Pr number property, only binary mixtures are currently supported", __FILE__, __LINE__); 
        } 

      }; 

  }; // class LaminarPrNo
}   // namespace Uintah

#endif
