#ifndef Uintah_Component_Arches_ChemHelper_h
#define Uintah_Component_Arches_ChemHelper_h

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>

namespace Uintah { 

  class ChemHelper { 
  
    public: 

      ChemHelper(){}; 
      ~ChemHelper(){}; 

      struct TableLookup{ 

        enum STATE { NEW, OLD };

        std::map<std::string, STATE> lookup;  
      
      };


    private: 
  
  }; // Class ChemHelper
} // Uintah namespace 

#endif 
