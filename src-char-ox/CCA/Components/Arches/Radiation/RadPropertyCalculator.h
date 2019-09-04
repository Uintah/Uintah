#ifndef Uintah_Component_Arches_RadPropertyCalculator_h
#define Uintah_Component_Arches_RadPropertyCalculator_h

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <vector>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>

#include <sci_defs/uintah_defs.h>

#ifdef HAVE_RADPROPS
#  include <radprops/AbsCoeffGas.h>
#endif

namespace Uintah { 

  class RadPropertyCalculator{ 

    public: 

      RadPropertyCalculator( const int _matl_index );

      ~RadPropertyCalculator();

      typedef std::vector<constCCVariable<double> > RadCalcSpeciesList; 

      /** @brief Problem setup **/ 
      void problemSetup( const ProblemSpecP& db ); 
      
      /** @brief Compute the properties/timestep **/ 
      void sched_compute_radiation_properties( const LevelP& level, SchedulerP& sched, const MaterialSet* matls, 
                                               const int time_substep, const bool doing_initialization ); 

      /** @brief see sched_compute_radiation_properties **/ 
      void compute_radiation_properties( const ProcessorGroup* pc, 
                                         const PatchSubset* patches, 
                                         const MaterialSubset* matls, 
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw, 
                                         const int time_substep, 
                                         const bool doing_initialization);

      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase() {}
          virtual ~PropertyCalculatorBase(){

          }

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  )=0; 
          virtual void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                     RadCalcSpeciesList species, constCCVariable<double>& mixT,  
                                     CCVariable<double>& abskg )=0; 


          virtual std::vector<std::string> get_sp() = 0;

          inline const VarLabel* get_abskg_label() { return _abskg_label; } 


          std::string get_abskg_name(){ return _abskg_name;}

          /** @brief This function sums in the particle contribution to the gas contribution **/ 
          template <class T> 
          void sum_abs( CCVariable<double>& absk_tot, T& abskp, const Patch* patch ){ 
            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              absk_tot[*iter] += abskp[*iter];  
              
            }
          }

          void setPressure(){
            ChemHelper& helper = ChemHelper::self();
            // Example on getting the table constants
            d_gasPressure=1.0; // in atm
            ChemHelper::TableConstantsMapType the_table_constants = helper.get_table_constants();
            if (the_table_constants !=nullptr){
              auto press_iter = the_table_constants->find("Pressure");
              if ( press_iter != the_table_constants->end() ){
            d_gasPressure=press_iter->second/101325.; // in atm
              }
            }
          }



        protected: 

          const VarLabel* _abskg_label;   // gas absorption coefficient
          double d_gasPressure;       // gas pressure in atm

          std::string _abskg_name; 



      };

      typedef std::vector<PropertyCalculatorBase*> CalculatorVec;
      CalculatorVec _all_calculators; 

      //______________________________________________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties();
          ~ConstantProperties();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }

        private: 
          double _abskg_value; 


      }; 
      //______________________________________________________________________
      //
      class specialProperties : public PropertyCalculatorBase  { 

        public: 
          specialProperties();
          ~specialProperties();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }

        private: 
          double _expressionNumber; 


      }; 
      //______________________________________________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston();
          ~BurnsChriston();
          bool problemSetup( const ProblemSpecP& db ); 
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }
        private: 
          double _value;
          Point _notSetMin;
          Point _notSetMax;
          Point _min;
          Point _max;
      }; 
      //______________________________________________________________________
      //
      class HottelSarofim : public PropertyCalculatorBase  { 

        public: 
          HottelSarofim();
          ~HottelSarofim();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp();

        private: 

          std::string _co2_name;                     ///< table name
          std::string _h2o_name;                     ///< table name 
          std::string _soot_name;                    ///< property name
          double d_opl;                              ///< optical length; 
      }; 

      class GauthamWSGG : public PropertyCalculatorBase  { 

        public: 
           GauthamWSGG();
          ~GauthamWSGG();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp();

        private: 

          std::string _mixMolecWeight;         
          std::vector<double> _sp_mw; 
          std::vector<std::vector<double> > _K ;     ///< gas absorption coeeficient / mol fraction of gas mixture
          std::vector<std::vector<double> > _C1;     ///< temperature dependence coefficient for weights (slope)
          std::vector<std::vector<double> > _C2;     ///< temperature dependence coefficient for weights (intercept)
          std::string _co2_name;                     ///< table name
          std::string _h2o_name;                     ///< table name 
          std::string _soot_name;                    ///< property name
          double d_opl;                              ///< optical length; 
      }; 

#ifdef HAVE_RADPROPS
      //______________________________________________________________________
      //
      class RadPropsInterface : public PropertyCalculatorBase  { 

        public: 
          RadPropsInterface();
          ~RadPropsInterface(); 
          bool problemSetup( const ProblemSpecP& db );
          void initialize_abskg( const Patch* patch,CCVariable<double>& abskg  ); 
          void compute_abskg( const Patch* patch, 
              constCCVariable<double>& VolFractionBC, 
              RadCalcSpeciesList species,  
              constCCVariable<double>& mixT, 
              CCVariable<double>& abskg); 

          std::vector<std::string> get_sp(){ return _species; }

        private: 

          RadProps::GreyGas*                      _gg_radprops;
          std::vector<std::string>                _species;               // to match the Arches varlabels
          std::vector<RadProps::RadiativeSpecies> _radprops_species; // for rad props
          std::string                             _mix_mol_weight_name; 
          std::vector<double>                     _sp_mw; 

      }; 
#endif


    private: 

      const int _matl_index; 
      std::string _temperature_name; 
      const VarLabel* _temperature_label; 


  }; 
} 

#endif



