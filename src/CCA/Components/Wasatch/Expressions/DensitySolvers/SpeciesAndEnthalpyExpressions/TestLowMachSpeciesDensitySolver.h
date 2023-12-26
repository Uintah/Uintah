/*
 * The MIT License
 *
<<<<<<< HEAD
 * Copyright (c) 2012-2023 The University of Utah
=======
<<<<<<< HEAD:src/Core/Grid/TaskStatus.h
 * Copyright (c) 1997-2019 The University of Utah
=======
 * Copyright (c) 2012-2023 The University of Utah
>>>>>>> master:src/CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/TestLowMachSpeciesDensitySolver.h
>>>>>>> master
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
#ifndef test_species_density_solver_h
#define test_species_density_solver_h

<<<<<<< HEAD
=======
<<<<<<< HEAD:src/Core/Grid/TaskStatus.h
#ifndef CORE_GRID_TASK_STATUS_H
#define CORE_GRID_TASK_STATUS_H

namespace Uintah {


  enum CallBackEvent
  {
      CPU      // <- normal CPU task, happens when a GPU enabled task runs on CPU
    , preGPU   // <- pre GPU kernel callback, happens before CPU->GPU copy (reserved, not implemented yet... )
    , GPU      // <- GPU kernel callback, happens after dw: CPU->GPU copy, kernel launch should be queued in this callback
    , postGPU  // <- post GPU kernel callback, happens after dw: GPU->CPU copy but before MPI sends.
  };

} //end namespace Uintah
#endif //#ifndef CORE_GRID_TASK_STATUS_H
=======
>>>>>>> master
#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error test code for the density solver for low-Mach species transport requires PoKiTT.
#endif

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>

namespace WasatchCore{

void test_low_mach_species_density_solver( Uintah::ProblemSpecP& params,
                                           GraphCategories& gc,
                                           std::set<std::string>& persistentFields );

} // namespace WasatchCore

#endif // test_species_density_solver_h
<<<<<<< HEAD
=======
>>>>>>> master:src/CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/TestLowMachSpeciesDensitySolver.h
>>>>>>> master
