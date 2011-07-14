#include "SetProcID.h"
#include "FieldAdaptor.h"

//-- Uintah includes --//
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>

namespace Wasatch{

  typedef Uintah::CCVariable<double> FieldT;

  //------------------------------------------------------------------

  SetProcID::SetProcID( Uintah::SchedulerP& sched,
                        const Uintah::PatchSet* patches,
                        const Uintah::MaterialSet* materials )
  {
    pid_ = Uintah::VarLabel::create( "rank",
                                     FieldT::getTypeDescription(),
                                     Uintah::IntVector(0,0,0) );
                                     
    Uintah::Task* task = scinew Uintah::Task( "set_rank", this, &SetProcID::set_rank );
    task->computes( pid_,
                    patches->getUnion(), Uintah::Task::NormalDomain,
                    materials->getUnion(), Uintah::Task::NormalDomain );
    sched->addTask( task, patches, materials );
  }

  //------------------------------------------------------------------

  SetProcID::~SetProcID()
  {
    Uintah::VarLabel::destroy( pid_ );
  }

  //------------------------------------------------------------------

  void SetProcID::set_rank( const Uintah::ProcessorGroup* const pg,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            Uintah::DataWarehouse* const oldDW,
                            Uintah::DataWarehouse* const newDW )
  {
    for( int ip=0; ip<patches->size(); ++ip ){
      const Uintah::Patch* const patch = patches->get(ip);

      for( int im=0; im<materials->size(); ++im ){
        const int material = materials->get(im);

        Uintah::CCVariable<double> rank;
        newDW->allocateAndPut( rank,
                               pid_,
                               material,
                               patch,
                               Uintah::Ghost::AroundCells,
                               1 );

        for( Uintah::CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ){
          rank[*iter] = pg->myrank();
        }
      } // materials
    } // patches
  }

  //------------------------------------------------------------------

} // namespace Wasatch
