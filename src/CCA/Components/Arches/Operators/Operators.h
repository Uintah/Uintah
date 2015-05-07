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

    void create_patch_operators( const LevelP& level, SchedulerP& sched,
                                       const MaterialSet* matls ); 

    void set_my_world( const ProcessorGroup* myworld ){ 
      _myworld = myworld; 
    };

    void delete_patch_set(); 

  private: 

    Operators(); 

    ~Operators(); 
    const ProcessorGroup* _myworld; 

    enum PatchsetSelector{
      USE_FOR_TASKS,
      USE_FOR_OPERATORS
    };

    /** \brief obtain the set of patches to operate on */
    const Uintah::PatchSet* get_patchset( const PatchsetSelector,
                                          const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched );

    std::map<int, Uintah::PatchSet*> _patches_for_operators; 

  };



}

#endif
