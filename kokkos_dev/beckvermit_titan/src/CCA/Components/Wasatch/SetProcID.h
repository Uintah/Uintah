/*
 * The MIT License
 *
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

  /**
   *  \class SetProcID
   *  \author James C. Sutherland
   *  \ingroup WasatchCore
   *
   *  \brief A simple task to run to set a CCVariable containing the
   *         processor rank.  This is typically only needed for
   *         diagnostic output.
   */
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
