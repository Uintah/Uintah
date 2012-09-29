/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "SolverAMG.h"

#include "util.h"
#include "Level.h"
#include "Patch.h"

#include <string>

using namespace std;

void
SolverAMG::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  dbg0 << "----------------------------------------------------" << "\n";
  dbg0 << "AMG setup phase" << "\n";
  dbg0 << "----------------------------------------------------" << "\n";
  HYPRE_BoomerAMGCreate(&_parSolver);
  HYPRE_BoomerAMGSetCoarsenType(_parSolver, 6);
  HYPRE_BoomerAMGSetStrongThreshold(_parSolver, 0.);
  HYPRE_BoomerAMGSetTruncFactor(_parSolver, 0.3);
  /*HYPRE_BoomerAMGSetMaxLevels(_parSolver, 4);*/
  HYPRE_BoomerAMGSetTol(_parSolver, 1.0e-06);
  HYPRE_BoomerAMGSetPrintLevel(_parSolver, 1);
  HYPRE_BoomerAMGSetPrintFileName(_parSolver, "sstruct.out.log");
  HYPRE_BoomerAMGSetMaxIter(_parSolver, 200);
  HYPRE_BoomerAMGSetup(_parSolver, _parA, _parB, _parX);
}

void
SolverAMG::solve(void)
{
  /*_____________________________________________________________________
    Function solveLinearSystem:
    Solve the linear system A*x = b using AMG. Includes parameter setup
    for the AMG solver phase.
    _____________________________________________________________________*/
  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  int                   time_index;  // Hypre Timer

  time_index = hypre_InitializeTiming("BoomerAMG Solve");
  hypre_BeginTiming(time_index);
  
  HYPRE_BoomerAMGSolve(_parSolver, _parA, _parB, _parX); // call AMG solver
  
  hypre_EndTiming(time_index);
  hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  /* Read results into Solver::_results */
  int numIterations = -1;
  HYPRE_BoomerAMGGetNumIterations(_parSolver, &numIterations);
  _results.numIterations = numIterations;
  HYPRE_BoomerAMGGetFinalRelativeResidualNorm(_parSolver,
                                              &_results.finalResNorm);
  
  HYPRE_BoomerAMGDestroy(_parSolver);

  /*-----------------------------------------------------------
   * Gather the solution vector
   *-----------------------------------------------------------*/
  dbg0 << "----------------------------------------------------" << "\n";
  dbg0 << "Gather the solution vector" << "\n";
  dbg0 << "----------------------------------------------------" << "\n";

  HYPRE_SStructVectorGather(_x);
} //end solve()
