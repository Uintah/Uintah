#ifndef Uintah_Component_Arches_RadPropertyCalculator_h
#define Uintah_Component_Arches_RadPropertyCalculator_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <sci_defs/uintah_defs.h>
#ifdef HAVE_RADPROPS
#  include <radprops/AbsCoeffGas.h>
#  include <radprops/RadiativeSpecies.h>
#  include <radprops/Particles.h>
#endif

namespace Uintah { 

  class RadPropertyCalculator{ 

    public: 

      RadPropertyCalculator();

      ~RadPropertyCalculator();

      typedef std::vector<constCCVariable<double> > RadCalcSpeciesList; 

      bool problemSetup( const ProblemSpecP& db ){ 

        if ( db->findBlock("property_calculator") ){ 

          std::string calculator_type; 
          ProblemSpecP db_pc = db->findBlock("property_calculator"); 

          db_pc->getAttribute("type", calculator_type); 

          if ( calculator_type == "constant" ){ 
            _calculator = scinew ConstantProperties(); 
          } else if ( calculator_type == "burns_christon" ){ 
            _calculator = scinew BurnsChriston(); 
          } else if ( calculator_type == "hottel_sarofim"){
            _calculator = scinew HottelSarofim(); 
          } else if ( calculator_type == "radprops" ){
#ifdef HAVE_RADPROPS
            _calculator = scinew RadPropsInterface(); 
#else
            throw InvalidValue("Error: You haven't configured with the RadProps library (try configuring with --enable-wasatch_3p and --with-boost=DIR.)",__FILE__,__LINE__);
#endif
          } else { 
            throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 
          } 

          bool complete; 
          complete = _calculator->problemSetup( db_pc );

          return complete; 

        } 

        return false; 

      };

      void compute( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 

      _calculator->computeProps( patch, VolFractionBC, species, mixT, abskg );

      };

      void compute( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                    double weights_scaling_constant, RadCalcSpeciesList weights, const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp ){

      _calculator->computePropsWithParticles( patch, VolFractionBC, species, size_scaling_constant, size, pT, weights_scaling_constant, weights, N, mixT, abskg, abskp ); 

      };

      inline std::vector<std::string> get_participating_sp(){ 

        return _calculator->get_sp(); 

      }

      bool does_scattering() { 

        return _calculator->does_scattering(); 

      } 

    private: 

      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase() {}; 
          virtual ~PropertyCalculatorBase(){};

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                     RadCalcSpeciesList species, constCCVariable<double>& mixT,  
                                     CCVariable<double>& abskg )=0; 
          virtual void computePropsWithParticles( const Patch* patch,
                                                  constCCVariable<double>& VolFractionBC,
                                                  RadCalcSpeciesList species,
                                                  double size_scaling_constant,
                                                  RadCalcSpeciesList size,
                                                  RadCalcSpeciesList pT,
                                                  double weights_scaling_constant,
                                                  RadCalcSpeciesList weight,
                                                  const int N,
                                                  constCCVariable<double>& mixT,
                                                  CCVariable<double>& abskg,
                                                  CCVariable<double>& abskp ) = 0;
          virtual std::vector<std::string> get_sp() = 0;
          virtual bool does_scattering() = 0;
      };
      //______________________________________________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties();
          ~ConstantProperties();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void computePropsWithParticles( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weight_scaling_constant, RadCalcSpeciesList weight, 
                                          const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp );
          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }
          bool does_scattering(){ return false; }

        private: 
          double _value; 
      }; 
      //______________________________________________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston();
          ~BurnsChriston();
          bool problemSetup( const ProblemSpecP& db ); 
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void computePropsWithParticles( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weight_scaling_constant, RadCalcSpeciesList weight, 
                                          const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp );
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
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );
          void computePropsWithParticles( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weight_scaling_constant, RadCalcSpeciesList weight, 
                                          const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp );
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
          void computeProps( const Patch* patch, 
              constCCVariable<double>& VolFractionBC, 
              RadCalcSpeciesList species,  
              constCCVariable<double>& mixT, 
              CCVariable<double>& abskg); 

          void computePropsWithParticles( const Patch* patch,  constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp );

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

      RadPropertyCalculator::PropertyCalculatorBase* _calculator;

  }; 
} 

#endif
