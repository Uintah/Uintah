
#ifndef Uintah_Component_Arches_RadPropertyCalculator_h
#define Uintah_Component_Arches_RadPropertyCalculator_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>

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

      void compute( const Patch* patch, CCVariable<double> abskg ){ 

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

      //__________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties() {}
          ~ConstantProperties() {}

          bool problemSetup( const ProblemSpecP& db ) {
              
            ProblemSpecP db_prop = db; 
            db_prop->getWithDefault("abskg",_value,1.0); 
            
            bool property_on = true; 

            return property_on; 
          };

          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 
            abskg.initialize(_value); 
          }; 

        private: 
          double _value; 
      }; 

      //__________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston() {}
          ~BurnsChriston() {}

          bool problemSetup( const ProblemSpecP& db ) { 
            ProblemSpecP db_prop = db; 
            db_prop->require("grid",grid);
             
            bool property_on = true; 
            return property_on; 
          };

          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 

            Vector Dx = patch->dCell(); 

            for (CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){ 
              IntVector c = *iter; 
              std::cout << abskg[c] << std::endl;
              abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (grid.x() - 1.0) /2.0) * Dx[0]) )
                              * ( 1.0 - 2.0 * fabs( ( c[1] - (grid.y() - 1.0) /2.0) * Dx[1]) )
                              * ( 1.0 - 2.0 * fabs( ( c[2] - (grid.z() - 1.0) /2.0) * Dx[2]) ) 
                              + 0.1;
            } 
          }; 

        private: 
          double _value; 
          IntVector grid; 
      }; 

      RadPropertyCalculator::PropertyCalculatorBase* _calculator;

  }; 
} 

#endif
