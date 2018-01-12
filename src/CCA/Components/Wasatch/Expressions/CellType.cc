/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

//-- Wasatch Includes --//
#include "CellType.h"
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

//-- Uintah Includes --//
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>

enum CellTypeEnum {
  FLOW = -1
  };

//------------------------------------------------------------------

CellType::CellType()
{
  cellTypeVarLabel_ = Uintah::VarLabel::create( WasatchCore::TagNames::self().celltype.name(),
                                                Uintah::CCVariable<int>::getTypeDescription() );
}

//------------------------------------------------------------------

CellType::~CellType()
{
  Uintah::VarLabel::destroy(cellTypeVarLabel_);
}

//------------------------------------------------------------------

void
CellType::schedule_compute_celltype (Uintah::Ray* rmcrt,
                                     const Uintah::PatchSet* const patches,
                                     const Uintah::MaterialSet* const materials,
                                     Uintah::SchedulerP& sched)
{
  // create the Uintah task to accomplish this.
  Uintah::Task* computeCellTypeTask = scinew Uintah::Task( "WasatchCore::compute_celltype", this, &CellType::compute_celltype, rmcrt );
  computeCellTypeTask->computes(cellTypeVarLabel_);
  sched->addTask( computeCellTypeTask, patches, materials );
}

//------------------------------------------------------------------

void CellType::compute_celltype(const Uintah::ProcessorGroup* const pg,
                                const Uintah::PatchSubset* const patches,
                                const Uintah::MaterialSubset* const materials,
                                Uintah::DataWarehouse* const oldDW,
                                Uintah::DataWarehouse* const newDW,
                                Uintah::Ray* rmcrt)
{
  typedef Uintah::CCVariable<int>       UintahField;
  for( int ip=0; ip<patches->size(); ++ip ){
    const Uintah::Patch* const patch = patches->get(ip);
    for( int im=0; im<materials->size(); ++im ){
      UintahField cellType;
      newDW->allocateAndPut( cellType, cellTypeVarLabel_, im, patch );
      cellType.initialize(FLOW);
      rmcrt->setBC<int, int>( cellType, cellTypeVarLabel_->getName(), patch, im );
    }
  }
}

//------------------------------------------------------------------

void CellType::schedule_carry_forward(const Uintah::PatchSet* const patches,
                                      const Uintah::MaterialSet* const materials,
                                      Uintah::SchedulerP& sched)
{
  Uintah::Task* tsk = scinew Uintah::Task( "WasatchCore::CellType::carry_forward", this, &CellType::carry_forward );
  tsk->requires(Uintah::Task::OldDW, cellTypeVarLabel_, Uintah::Ghost::None, 0);
  tsk->computes(cellTypeVarLabel_);
  sched->addTask( tsk, patches, materials );
}

//------------------------------------------------------------------

void
CellType::carry_forward(const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW)
{
  newDW->transferFrom(oldDW, cellTypeVarLabel_, patches, materials);
}

//------------------------------------------------------------------
