/**
 *  \file   ArchesBCHelper.h
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2015 The University of Utah
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

#ifndef ArchesBC_Helper_h
#define ArchesBC_Helper_h

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <CCA/Components/Wasatch/BCHelper.h>

// forward declarations
namespace Uintah{

  class ProcessorGroup;
  class DataWarehouse;
  class Arches;

  /**
   *  \class  ArchesBCHelper
   *  \brief  This class provides support for boundary conditions.
   */

  class ArchesBCHelper: public WasatchCore::BCHelper
  {
  public:

    ArchesBCHelper( const Uintah::LevelP& level,
                    Uintah::SchedulerP& sched,
                    const Uintah::MaterialSet* const materials );
    ~ArchesBCHelper();

  private:

  };

} /* namespace Arches */
#endif /* ArchesBC_Helper_H */
