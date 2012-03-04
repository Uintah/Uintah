
#ifndef Uintah_Component_Arches_RadPropertyCalculator_h
#define Uintah_Component_Arches_RadPropertyCalculator_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

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
            notSet = Point(-9,-9,-9);
            min    = notSet;
            max    = notSet;
          }
          ~BurnsChriston() {}
          
          //__________________________________
          //
          bool problemSetup( const ProblemSpecP& db ) { 
            ProblemSpecP db_prop = db;
            
            db_prop->get("min", min);  // optional
            db_prop->get("max", max);
            
            bool property_on = true; 
            return property_on; 
          };
          
          //__________________________________
          //
          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 
            
            BBox domain(min,max);
            
            // if the user didn't specify the min and max 
            // use the grid's domain
            if( min == notSet  &&  max == notSet ){
              const Level* level = patch->getLevel();
              GridP grid  = level->getGrid();
              grid->getInteriorSpatialRange(domain);
            }

            Point midPt((max - min)/2 + min);
            
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
          Point notSet;
          Point min;
          Point max;
           
      }; 

      RadPropertyCalculator::PropertyCalculatorBase* _calculator;

  }; 
} 

#endif
