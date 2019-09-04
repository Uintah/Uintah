#ifndef Uintah_Component_Arches_ChemHelper_h
#define Uintah_Component_Arches_ChemHelper_h

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>

namespace Uintah {

  class ChemHelper {

    public:

      ChemHelper(){};
      ~ChemHelper(){};

      enum STATE { NEW, OLD };

      typedef std::map<std::string, double>* TableConstantsMapType;

      static ChemHelper& self(){
        static ChemHelper s;
        return s;
      }

      inline void add_lookup_species( std::string name, STATE state=NEW ) {

        if ( name == "" ){
          throw InvalidValue(
            "Error: Passing an empty string for table lookup.", __FILE__, __LINE__ );
        }

        if ( state == NEW ){
          model_req_species.push_back( name );
        } else {
          model_req_old_species.push_back( name );
        }

      }

      void set_table_constants( std::map<std::string, double>* constants ){
        m_table_constants = constants;
      }

      TableConstantsMapType get_table_constants(){ return m_table_constants; }

      std::vector<std::string> model_req_species;
      std::vector<std::string> model_req_old_species;

    private:

      TableConstantsMapType m_table_constants;

  }; // Class ChemHelper
} // Uintah namespace

#endif
