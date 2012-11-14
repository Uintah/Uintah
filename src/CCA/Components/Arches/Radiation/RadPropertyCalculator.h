
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
            std::cout << "going to allocated a radprops interface " << std::endl;
            _calculator = scinew RadPropsInterface(); 
#else
            throw InvalidValue("Error: You haven't configured with RADPROPS!.",__FILE__,__LINE__);
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

      void compute( const Patch* patch, CCVariable<double>& abskg ){ 

        _calculator->computeProps( patch, abskg );

      };

    private: 

      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase() {} 
          virtual ~PropertyCalculatorBase() {}

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void computeProps( const Patch* patch, CCVariable<double>& abskg )=0;  // for now only assume abskg
      };
#ifdef HAVE_RADPROPS
      //______________________________________________________________________
      //
      class RadPropsInterface : public PropertyCalculatorBase  { 

        public: 
          RadPropsInterface() {}
          ~RadPropsInterface() {}
          
          //__________________________________
          //
          bool problemSetup( const ProblemSpecP& db ) {

            if ( db->findBlock( "grey_gas" ) ){

              ProblemSpecP db_gg = db->findBlock( "grey_gas" ); 
              std::vector<RadiativeSpecies> species; 

              //look for all the participating species: 
              for ( ProblemSpecP db_sp = db_gg->findBlock("species"); db_sp != 0; db_sp = db_sp->findNextBlock("species") ){ 
                std::string current_sp; 
                db_sp->getAttribute("label",current_sp); 
                RadiativeSpecies radprop_sp = species_enum( current_sp );
                species.push_back( radprop_sp );  
              }  

              //allocate gray gas object: 
              _gg_radprops = scinew GreyGas( species ); 

            } else { 

              throw InvalidValue( "Error: Only grey gas properties are available at this time.",__FILE__,__LINE__);

            }
              
            
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 

            // below is just a placeholder.  
            // here we need to:  
            // 1) pass into this method the participating species
            // (the rest of this should be in a grid loop )
            // 2) convert them to mol fractions
            // 3) package them into the molFrac vector
            // 4) call the mixture_coeffs function to get back the abskg
            // 5) assign abskg[c] to the value out of the lookup 

            double plankCff = 0.0;
            double rossCff  = 0.0; 
            double effCff   = 0.0; 
            std::vector<double> molFrac; 
            double T        = 298; 

            _gg_radprops->mixture_coeffs( planckCff, rossCff, effCff, mixMoleFrac, TMix );

            

          }; 

        private: 

          AbsCoeffGas::GreyGas* _gg_radprops; 
      }; 
#endif
      //______________________________________________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties() {}
          ~ConstantProperties() {}
          
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
          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 
            abskg.initialize(_value); 
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
          }
          ~BurnsChriston() {}
          
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
          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 
            
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
