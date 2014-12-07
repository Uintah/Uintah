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

#include <sci_defs/uintah_defs.h>

#ifdef HAVE_RADPROPS
#  include <radprops/AbsCoeffGas.h>
#  include <radprops/RadiativeSpecies.h>
#  include <radprops/Particles.h>
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

            VarLabel::destroy(_abskg_label); 
            if ( _local_abskp ) {
              VarLabel::destroy(_abskp_label); 
            }
            if ( _use_scatkt ) {
              VarLabel::destroy(_scatkt_label); 
            }
          }

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                     RadCalcSpeciesList species, constCCVariable<double>& mixT,  
                                     CCVariable<double>& abskg )=0; 
          virtual void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& abskp )=0;

          virtual void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& scatkt )=0;

          virtual std::vector<std::string> get_sp() = 0;
          virtual bool does_scattering() = 0;

          inline const VarLabel* get_abskg_label() { return _abskg_label; } 
          inline const VarLabel* get_abskp_label() { return _abskp_label; }
          inline const VarLabel* get_scatkt_label(){ return _scatkt_label;}
          inline       bool      has_abskp_local() { return _local_abskp; }
          inline       bool      use_abskp()       { return _use_abskp;   }
          inline       bool      use_scatkt()       { return _use_scatkt;   }

          /** @brief Matches label names to labels **/ 
          inline void resolve_labels(){

            if ( _use_abskp && !_local_abskp ){ 
              const VarLabel* temp = VarLabel::find(_abskp_name); 
              if ( temp == 0 ){ 
                throw ProblemSetupException("Error: Could not find the abskp label.",__FILE__, __LINE__);
              } else { 
                _abskp_label = temp; 
              }
            }

            if ( _use_scatkt ){ 
              const VarLabel* temp = VarLabel::find(_scatkt_name); 
              if ( temp == 0 ){ 
                throw ProblemSetupException("Error: Could not find the scattering coefficient (scatkt) label.",__FILE__, __LINE__);
              } else { 
                _scatkt_label = temp; 
              }
            }
          } 

          std::string get_abskg_name(){ return _abskg_name;}
          std::string get_abskp_name(){ return _abskp_name;}
          std::string get_scatkt_name(){return _scatkt_name;}

          /** @brief This function sums in the particle contribution to the gas contribution **/ 
          template <class T> 
          void sum_abs( CCVariable<double>& absk_tot, T& abskp, const Patch* patch ){ 
            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              absk_tot[*iter] += abskp[*iter];  
              
            }
          }

        protected: 

          const VarLabel* _abskg_label;   // gas absorption coefficient
          const VarLabel* _abskp_label;   // particle absorption coefficient
          const VarLabel* _scatkt_label;  // particle scattering coefficient

          std::string _abskg_name; 
          std::string _abskp_name; 
          std::string _scatkt_name; 

          bool _local_abskp; 
          bool _use_abskp; 

          bool _use_scatkt;  // local not needed for scattering, because it is always local as of 11-2014

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
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn,   CCVariable<double>& abskp );
          void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& scatkt );
          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }
          bool does_scattering(){ return false; }

        private: 
          double _abskg_value; 
          double _abskp_value; 


      }; 
      //______________________________________________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston();
          ~BurnsChriston();
          bool problemSetup( const ProblemSpecP& db ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn,   CCVariable<double>& abskp );
          void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& scatkt );
          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }
          bool does_scattering(){ return false; }
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
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn,   CCVariable<double>& abskp );
          void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& scatkt );
          std::vector<std::string> get_sp();
          bool does_scattering(); 

        private: 

          std::vector<std::string> _the_species;     ///< list of species
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
          void compute_abskg( const Patch* patch, 
              constCCVariable<double>& VolFractionBC, 
              RadCalcSpeciesList species,  
              constCCVariable<double>& mixT, 
              CCVariable<double>& abskg); 

          void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn,  CCVariable<double>& abskp );
          void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, CCVariable<double>& scatkt );

          std::vector<std::string> get_sp(){ return _species; }
          bool does_scattering(){ return _does_scattering; }

        private: 

          GreyGas* _gg_radprops; 
          ParticleRadCoeffs* _part_radprops; 
          std::vector<std::string> _species;               // to match the Arches varlabels
          std::vector<RadiativeSpecies> _radprops_species; // for rad props
          std::string _mix_mol_weight_name; 
          std::vector<double> _sp_mw; 
          bool _does_scattering; 
          bool _p_planck_abskp; 
          bool _p_ros_abskp; 

      }; 
#endif

    private: 

      const int _matl_index; 
      std::string _temperature_name; 
      const VarLabel* _temperature_label; 


  }; 
} 

#endif
