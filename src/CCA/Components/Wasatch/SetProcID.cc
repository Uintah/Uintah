/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
