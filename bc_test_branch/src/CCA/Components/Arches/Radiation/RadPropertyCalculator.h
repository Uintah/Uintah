
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
#include <radprops/AbsCoeffGas.h>
#include <radprops/RadiativeSpecies.h>
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

      void compute( const Patch* patch, RadCalcSpeciesList species, CCVariable<double>& abskg ){ 

        _calculator->computeProps( patch, species, abskg );

      };

      inline std::vector<std::string> get_participating_sp(){ 
        return _calculator->get_sp(); 
      }

    private: 

      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase() {}; 
          virtual ~PropertyCalculatorBase(){};

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void computeProps( const Patch* patch, RadCalcSpeciesList species, CCVariable<double>& abskg )=0;  // for now only assume abskg
          virtual std::vector<std::string> get_sp()=0;
      };
#ifdef HAVE_RADPROPS
      //______________________________________________________________________
      //
      class RadPropsInterface : public PropertyCalculatorBase  { 

        public: 
          RadPropsInterface() {};
          ~RadPropsInterface() {};
          
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
              _species.push_back(_mix_mol_weight_name); 

              // NOTE: this requires that the table names match the RadProps name.  This is, in general, a pretty 
              // bad assumption.  Need to make this more robust later on...
              //
              for ( std::vector<RadiativeSpecies>::iterator iter = _radprops_species.begin(); iter != _radprops_species.end(); iter++){
                std::string which_species = species_name( *iter ); 
                _species.push_back( which_species ); 

                if ( which_species == "CO2" ){ 
                  _sp_mw.push_back(44.0);
                } else if ( which_species == "H2O" ){ 
                  _sp_mw.push_back(18.0); 
                } else if ( which_species == "CO" ){ 
                  _sp_mw.push_back(28.0); 
                } else if ( which_species == "NO" ){
                  _sp_mw.push_back(30.0); 
                } else if ( which_species == "OH" ){
                  _sp_mw.push_back(17.0); 
                } 
              }

            } else { 

              throw InvalidValue( "Error: Only grey gas properties are available at this time.",__FILE__,__LINE__);

            }

            //need smarter return? 
            //or no return at all? 
            return true; 
            
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, RadCalcSpeciesList species, CCVariable<double>& abskg ){ 

            int N = species.size(); 

            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              IntVector c = *iter; 

              double plankCff = 0.0;
              double rossCff  = 0.0; 
              double effCff   = 0.0; 
              std::vector<double> mol_frac; 
              double T        = 298;

              //convert mass frac to mol frac
              for ( int i = 2; i < N; i++ ){ 
                double value = (species[i])[c] * _sp_mw[i-1] * (species[1])[c];
                //              ^^species^^^^    ^^MW^^^^^^    ^^^MIX MW^^^^^^^
                mol_frac.push_back(value); 
              } 

              _gg_radprops->mixture_coeffs( plankCff, rossCff, effCff, mol_frac, T );

              abskg[c] = effCff; //need to generalize this to the other coefficients  

            }

          }; 

          std::vector<std::string> get_sp(){
            return _species; 
          };

        private: 

          GreyGas* _gg_radprops; 
          std::vector<std::string> _species;               // to match the Arches varlabels
          std::vector<RadiativeSpecies> _radprops_species; // for rad props
          std::string _mix_mol_weight_name; 
          std::vector<double> _sp_mw; 

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
          void computeProps( const Patch* patch, RadCalcSpeciesList species, CCVariable<double>& abskg ){ 
            abskg.initialize(_value); 
          }; 

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          };

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
          void computeProps( const Patch* patch, RadCalcSpeciesList species, CCVariable<double>& abskg ){ 
            
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

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          };

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
