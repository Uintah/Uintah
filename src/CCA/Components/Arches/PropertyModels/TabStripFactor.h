#ifndef Uintah_Component_Arches_TabStripFactor_h
#define Uintah_Component_Arches_TabStripFactor_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

/** 
* @class  TabStripFactor
* @author Jeremy Thornock
* @date   Aug 2011
* 
* @brief Computes the stripping factor of CxHy using tabulated values of CO2.  
*
* @details This class computes the stripping factor of any hydrocarbon defined
*          by CxHy where x and y are integers.  The stipping factor is defined
*          as the local amount of CxHy reacted divided by the the local amount 
*          available.  
*
*          Note that the Westbrook-Dryer model also computes a stripping factor
*          based on resolved scale CxHy transport.  This class assumes that the 
*          reactions are too fast and therefore the stripping factor must be 
*          backed out from the tabulated CO2.  
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label="my_model" type="tab_strip_factor">
*       <X>DOUBLE</X>                  <!-- number of carbon atoms --> 
*       <Y>DOUBLE</Y>                  <!-- number of hydrogen atoms --> 
*       <fuel_mass_fraction>DOUBLE</fuel_mass_fraction>          <!-- mass fraction of hydrocarbon at f=1 --> 
*       <co2_label>STRING</co2_label>  <!-- label name for co2 --> 
*       <ch4_label>STRING</ch4_label>  <!-- label name for ch4 --> 
*       <mix_frac_label>STRING</mix_frac_label>      <!-- label name for mixture fraction --> 
*       <small>DOUBLE</small>          <!-- an optional value of a small number --> 
*     </model>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class TabStripFactor : public PropertyModelBase {

    public: 

      TabStripFactor( std::string prop_name, SimulationStateP& shared_state );
      ~TabStripFactor(); 

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

          TabStripFactor* build()
          { return scinew TabStripFactor( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      double _X;                            ///< Number of carbon atoms in the hydrocarbon 
      double _Y;                            ///< Number of hydrogen atoms in the hydrcarbon
      double _M_CO2;                        ///< Molecular weight of CO2
      double _M_HC;                         ///< MOlecular weight of hydrocarbon
      double _HC_F1;                        ///< Mass fraction of hydrocarbon at f = 1
      double _small;                        ///< A small number for denominator (default = 1e-10)
      std::string _co2_label;               ///< Name of the CO2 label (usually CO2 for Tabprops or co2IN for old table)
      std::string _ch4_label;               ///< Name of the CH4 label
      std::string _f_label;                 ///< Name of the mixture fraction label

  }; // class TabStripFactor
}   // namespace Uintah

#endif
