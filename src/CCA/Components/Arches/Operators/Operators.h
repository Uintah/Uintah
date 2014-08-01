#ifndef Uintah_Component_Arches_Operators_h
#define Uintah_Component_Arches_Operators_h

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

namespace Uintah { 

  class Operators{ 

  public: 

    static Operators& self(); 

    struct PatchInfo{

      SpatialOps::OperatorDatabase _sodb;

    };

    typedef std::map<int, PatchInfo> PatchInfoMap; 

    PatchInfoMap patch_info_map; 

  private: 

    Operators(); 

    ~Operators(); 


  };
}

#endif
