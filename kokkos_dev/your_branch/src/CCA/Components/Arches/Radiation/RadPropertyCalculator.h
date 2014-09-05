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

      RadPropertyCalculator(){
        _calculator = 0; 
      };

      ~RadPropertyCalculator(){
        delete _calculator; 
      };

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
          } else if ( calculator_type == "radprops" ){
#ifdef HAVE_RADPROPS
            _calculator = scinew RadPropsInterface(); 
#else
            throw InvalidValue("Error: You haven't configured with the RadProps library (try configuring with wasatch3p.)",__FILE__,__LINE__);
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

      const bool does_scattering(){ 

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
          virtual const bool does_scattering() = 0;
      };
#ifdef HAVE_RADPROPS
      //______________________________________________________________________
      //
      class RadPropsInterface : public PropertyCalculatorBase  { 

        public: 
          RadPropsInterface() 
          {
            _gg_radprops   = 0;
            _part_radprops = 0; 
            _p_ros_abskp  = false; 
            _p_planck_abskp = false; 
          }
          ~RadPropsInterface() {
          
            if ( _gg_radprops != 0 ) 
              delete _gg_radprops; 

            if ( _part_radprops != 0 ) 
              delete _part_radprops; 

          }
          
          //__________________________________
          //
          bool problemSetup( const ProblemSpecP& db ) {

            if ( db->findBlock( "grey_gas" ) ){

              ProblemSpecP db_gg = db->findBlock( "grey_gas" );

              db_gg->getWithDefault("mix_mol_w_label",_mix_mol_weight_name,"mixture_molecular_weight"); 
              std::string inputfile;
              db_gg->require("inputfile",inputfile); 

              //allocate gray gas object: 
              _gg_radprops = scinew GreyGas( inputfile ); 

              //get list of species: 
              _radprops_species = _gg_radprops->speciesGG(); 

              // mixture molecular weight will always be the first entry 
              // Note that we will assume the table value is the inverse
              _species.insert(_species.begin(), _mix_mol_weight_name);

              // NOTE: this requires that the table names match the RadProps name.  This is, in general, a pretty 
              // bad assumption.  Need to make this more robust later on...
              int total_sp = _radprops_species.size(); 
              for ( int i = 0; i < total_sp; i++ ){ 
                std::string which_species = species_name( _radprops_species[i] ); 
                _species.push_back( which_species ); 

                //entering the inverse here for convenience 
                if ( which_species == "CO2" ){ 
                  _sp_mw.push_back(1.0/44.0);
                } else if ( which_species == "H2O" ){ 
                  _sp_mw.push_back(1.0/18.0); 
                } else if ( which_species == "CO" ){ 
                  _sp_mw.push_back(1.0/28.0); 
                } else if ( which_species == "NO" ){
                  _sp_mw.push_back(1.0/30.0); 
                } else if ( which_species == "OH" ){
                  _sp_mw.push_back(1.0/17.0); 
                } 
              } 

            }else { 

              throw InvalidValue( "Error: Only grey gas properties are available at this time.",__FILE__,__LINE__);

            }

            // For particles: 
            _does_scattering = false; 
            if ( db->findBlock( "particles" ) ){ 

                ProblemSpecP db_p = db->findBlock( "particles" ); 

                double real_part = 0; 
                double imag_part = 0; 
                db_p->require( "complex_ir_real", real_part ); 
                db_p->require( "complex_ir_imag", imag_part ); 

                std::string which_model = "none"; 
                db_p->require( "model_type", which_model );
                if ( which_model == "planck" ){ 
                  _p_planck_abskp = true; 
                } else if ( which_model == "rossland" ){ 
                  _p_ros_abskp = true; 
                } else { 
                  throw InvalidValue( "Error: Particle model not recognized.",__FILE__,__LINE__);
                }   

                std::complex<double> complex_ir( real_part, imag_part ); 

                _part_radprops = scinew ParticleRadCoeffs( complex_ir ); 

                _does_scattering = true; 

              }
            //need smarter return? 
            //or no return at all? 
            return true; 
            
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species,  constCCVariable<double>& mixT, CCVariable<double>& abskg){ 

            int N = species.size(); 

            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              IntVector c = *iter; 

              double plankCff = 0.0;
              double rossCff  = 0.0; 
              double effCff   = 0.0; 
              std::vector<double> mol_frac; 
              double T        = mixT[c];
              double VolFraction = VolFractionBC[c];

              //convert mass frac to mol frac
              for ( int i = 1; i < N; i++ ){ 
                // Note that the species molecular weights and mixture molecular weights 
                // are actually the inverse 
                // as set in the problemsetup function.
                // Also note that the mixture molecular weight arriving from the table
                // is assumed to be the inverse mixture molecular weight

                double value = (species[0])[c]==0 ? 0.0 : (species[i])[c] * _sp_mw[i-1] * 1.0 / (species[0])[c];
                //                                          ^^species^^^^    ^^1/MW^^^^^  ^^^^^MIX MW^^^^^^^^^^
                
                if ( value < 0 ){ 
                  throw InvalidValue( "Error: For some reason I am getting negative mol fractions in the scattering portion of radiation property calculator.",__FILE__,__LINE__);
                } 

                mol_frac.push_back(value); 

              } 

              if ( VolFraction > 1.e-16 ){

                _gg_radprops->mixture_coeffs( plankCff, rossCff, effCff, mol_frac, T );

                abskg[c] = effCff; //need to generalize this to the other coefficients

              } else { 

                abskg[c] = 0.0; 

              }
            }

          };

        void computePropsWithParticles( const Patch* patch,  constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                          const int Nqn, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp ){
            int N = species.size(); 

            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              IntVector c = *iter; 

              double plankCff = 0.0;
              double rossCff  = 0.0; 
              double effCff   = 0.0; 
              std::vector<double> mol_frac; 
              double T        = mixT[c];
              double VolFraction = VolFractionBC[c];
              double unscaled_weight;
              double unscaled_size;

              //convert mass frac to mol frac
              for ( int i = 1; i < N; i++ ){ 

                double value = (species[0])[c]==0.0 ? 0.0 : (species[i])[c] * _sp_mw[i-1] * 1.0/(species[0])[c];
                //                                           ^^species^^^^    ^^1/MW^^^^^   ^^^^^MIX MW^^^^^^^^

                if(value<0){
                  throw InvalidValue( "Error: For some reason I am getting negative mol fractions in the radiation property calculator.",__FILE__,__LINE__);
                }

                mol_frac.push_back(value); 

              } 

              if ( VolFraction > 1.e-16 ){

                _gg_radprops->mixture_coeffs( plankCff, rossCff, effCff, mol_frac, T );

                abskg[c] = effCff; //need to generalize this to the other coefficients
              
                //now compute the particle values: 
                abskp[c] = 0.0; 
                for ( int i = 0; i < Nqn; i++ ){ 

                  unscaled_weight = (weights[i])[c]*weights_scaling_constant;
                  unscaled_size = (size[i])[c]*size_scaling_constant/(weights[i])[c];

                  if ( _p_planck_abskp ){ 

                    double abskp_i = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
                    abskp[c] += abskp_i * unscaled_weight; 
                    
                  } else if ( _p_ros_abskp ){ 

                    double abskp_i =  _part_radprops->ross_abs_coeff( unscaled_size, (pT[i])[c] );
                    abskp[c] += abskp_i * unscaled_weight; 

                  } 
                }

                abskg[c] += abskp[c]; 

              }else{    

                abskp[c] = 0.0;
                abskg[c] = 0.0;

              }
            }
          };

          std::vector<std::string> get_sp(){
            return _species; 
          };


          const bool does_scattering(){ return _does_scattering; }; 

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
      //______________________________________________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties() {};
          ~ConstantProperties() {};
          
          //__________________________________
          //
          bool problemSetup( const ProblemSpecP& db ) {
              
            ProblemSpecP db_prop = db; 
            db_prop->getWithDefault("abskg",_value,1.0); 
            
            bool property_on = true; 

            return property_on; 
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 
            abskg.initialize(_value); 
          }; 

          void computePropsWithParticles( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weight_scaling_constant, RadCalcSpeciesList weight, 
                                          const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp ){

            throw InvalidValue( "Error: No particle properties implemented for constant radiation properties.",__FILE__,__LINE__);

          };

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          };

          const bool does_scattering(){ return false; }; 

        private: 
          double _value; 
      }; 

      //______________________________________________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston() {
            _notSetMin = Point(SHRT_MAX, SHRT_MAX, SHRT_MAX);
            _notSetMax = Point(SHRT_MIN, SHRT_MIN, SHRT_MIN);
          };
          ~BurnsChriston() {};
          
          //__________________________________
          //
          bool problemSetup( const ProblemSpecP& db ) { 
            ProblemSpecP db_prop = db;
            
            db_prop->getWithDefault("min", _min, _notSetMin);  // optional
            db_prop->getWithDefault("max", _max, _notSetMax);
            
            // bulletproofing  min & max must be set
            if( ( _min == _notSetMin && _max != _notSetMax) ||
                ( _min != _notSetMin && _max == _notSetMax) ){
              ostringstream warn;
              warn << "\nERROR:<property_calculator type=burns_christon>\n "
                   << "You must specify both a min: "<< _min << " & max point: "<< _max <<"."; 
              throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
            }
            
            bool property_on = true; 
            return property_on; 
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 
            
            BBox domain(_min,_max);
            
            // if the user didn't specify the min and max 
            // use the grid's domain
            if( _min == _notSetMin  ||  _max == _notSetMax ){
              const Level* level = patch->getLevel();
              GridP grid  = level->getGrid();
              grid->getInteriorSpatialRange(domain);
              _min = domain.min();
              _max = domain.max();
            }
            
            Point midPt((_max - _min)/2 + _min);
            
            for (CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){ 
              IntVector c = *iter; 
              Point pos = patch->getCellPosition(c);
              
              if(domain.inside(pos)){
                abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( pos.x() - midPt.x() ) )
                                * ( 1.0 - 2.0 * fabs( pos.y() - midPt.y() ) )
                                * ( 1.0 - 2.0 * fabs( pos.z() - midPt.z() ) ) 
                                + 0.1;
              }
            } 
          }; 

          void computePropsWithParticles( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
                                          double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weight_scaling_constant, RadCalcSpeciesList weight, 
                                          const int N, constCCVariable<double>& mixT, CCVariable<double>& abskg, CCVariable<double>& abskp ){

            throw InvalidValue( "Error: No particle properties implemented for Burns/Christon radiation properties.",__FILE__,__LINE__);
          };

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          };

          const bool does_scattering(){ return false; }; 

        private: 
          double _value;
          Point _notSetMin;
          Point _notSetMax;
          Point _min;
          Point _max;
      }; 

      RadPropertyCalculator::PropertyCalculatorBase* _calculator;

  }; 
} 

#endif
