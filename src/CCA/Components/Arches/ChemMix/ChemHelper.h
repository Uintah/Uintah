#ifndef Uintah_Component_Arches_ChemHelper_h
#define Uintah_Component_Arches_ChemHelper_h

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>

namespace Uintah {

  class ChemHelper {

    public:

      ChemHelper(){};
      ~ChemHelper(){};

      enum STATE { NEW, OLD };

      static ChemHelper& self(){
        static ChemHelper s;
        return s;
      }

      inline void add_lookup_species( std::string name, STATE state=NEW ) {

        if ( state == NEW ){
          model_req_species.push_back( name );
        } else {
          model_req_old_species.push_back( name );
        }

      }

      std::vector<std::string> model_req_species;
      std::vector<std::string> model_req_old_species;

  }; // Class ChemHelper
} // Uintah namespace

#endif
