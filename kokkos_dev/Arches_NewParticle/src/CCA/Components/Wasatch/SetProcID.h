#ifndef Wasatch_SetProcID_h
#define Wasatch_SetProcID_h

//-- Uintah Framework Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

// forward declarations
namespace Uintah{
  class DataWarehouse;
  class ProcessorGroup;
  class Task;
  class VarLabel;
}

namespace Wasatch{
  class SetProcID
  {
    Uintah::VarLabel* pid_;
    void set_rank( const Uintah::ProcessorGroup* const pg,
                   const Uintah::PatchSubset* const patches,
                   const Uintah::MaterialSubset* const materials,
                   Uintah::DataWarehouse* const oldDW,
                   Uintah::DataWarehouse* const newDW );
  public:
    SetProcID( Uintah::SchedulerP& sched,
               const Uintah::PatchSet* patches,
               const Uintah::MaterialSet* materials );
    ~SetProcID();
  };
}


#endif // Wasatch_SetProcID_h
