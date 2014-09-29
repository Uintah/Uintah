#ifndef Uintah_Component_Arches_Operators_h
#define Uintah_Component_Arches_Operators_h

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/stencil/StencilBuilder.h>
#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>

namespace Uintah { 

  class Operators{ 

  public: 

    static Operators& self(); 

    struct PatchInfo{

      SpatialOps::OperatorDatabase _sodb;

    };

    typedef std::map<int, PatchInfo> PatchInfoMap; 

    PatchInfoMap patch_info_map; 

    void sched_create_patch_operators( const LevelP& level, SchedulerP& sched,
                                       const MaterialSet* matls ); 

  private: 

    Operators(); 

    ~Operators(); 

    void create_patch_operators( const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);


  };


}

#endif
