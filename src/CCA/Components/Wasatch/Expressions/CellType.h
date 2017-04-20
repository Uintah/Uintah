/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef CellType_h
#define CellType_h

#include <iostream>
#include <string>
#include <list>

#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/SchedulerP.h>

// forward declarations
namespace Uintah{
  class ProcessorGroup;
  class VarLabel;
  class DataWarehouse;
  class Ray;
}

/**
 *  \class 	   CellType
 *  \ingroup   Expressions
 *  \author 	 Tony Saad
 *  \date 	   November, 2013
 *
 *  \brief Provides a Uintah-task-based approach to setting the celltype.
 */
class CellType
{
public:
  
  CellType();
  ~CellType();
  
  void schedule_compute_celltype(Uintah::Ray* rmcrt,
                                 const Uintah::PatchSet* const patches,
                                 const Uintah::MaterialSet* const materials,
                                 Uintah::SchedulerP& sched);
  
  void schedule_carry_forward(const Uintah::PatchSet* const patches,
                                 const Uintah::MaterialSet* const materials,
                                 Uintah::SchedulerP& sched);

private:
  Uintah::VarLabel* cellTypeVarLabel_;

  void compute_celltype(const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW,
                        Uintah::Ray* rmcrt_);

  void carry_forward(const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW);
};

#endif /* defined(CellType_h) */

