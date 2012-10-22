/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include "SolverFAC.h"

#include "util.h"
#include "Level.h"
#include "Patch.h"
#include <string>

using namespace std;

void
SolverFAC::initializeData(const Hierarchy& hier,
                          const HYPRE_SStructGrid& grid)
{
  /* Initialize arrays needed by Hypre FAC */
  const Counter numLevels = hier._levels.size();
  const Counter numDims = hier._param->numDims;
  _pLevel          = hypre_TAlloc(int , numLevels);    
  _refinementRatio = hypre_TAlloc(hypre_Index, numLevels);
  for (Counter level = 0; level < numLevels; level++) {
    _pLevel[level] = level;   // part ID of this level
    for (Counter d = 0; d < numDims; d++) { // Assuming numDims = 3
      _refinementRatio[level][d] = hier._levels[level]->_refRat.getData()[d];
    }
  }

  Solver::initializeData(hier, grid); // Do the rest of the generic inits
}

void
SolverFAC::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  dbg << "----------------------------------------------------" << "\n";
  dbg << "FAC setup phase" << "\n";
  dbg << "----------------------------------------------------" << "\n";

  /* FAC Solver. Prepare FAC operator hierarchy using Galerkin coarsening
     with black-box interpolation, on the original meshes */
  int time_fac_rap = hypre_InitializeTiming("fac rap");
  hypre_BeginTiming(time_fac_rap);
  hypre_AMR_RAP(_A, _refinementRatio, &_facA);
  hypre_EndTiming(time_fac_rap);
  hypre_PrintTiming("fac rap", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_fac_rap);
  hypre_ClearTiming();
}

void
SolverFAC::solve(void)
{
  // TODO - solve

  /*_____________________________________________________________________
    Function solveLinearSystem:
    Solve the linear system A*x = b. The result is returned into x.
    Solvers include FAC and AMG.
    _____________________________________________________________________*/
  const int numLevels = _param->numLevels;

  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  HYPRE_SStructSolver   solver;
  int                   n_pre, n_post;

  /* Timers, debugging flags */
  int                   time_index;

  /*-------------- FAC Solver -----------------*/
  n_pre  = _refinementRatio[numLevels-1][0]-1;
  n_post = _refinementRatio[numLevels-1][0]-1;

  /* n_pre+= n_post;*/
  /* n_post= 0;*/

  time_index = hypre_InitializeTiming("FAC Setup");
  hypre_BeginTiming(time_index);

  HYPRE_SStructFACCreate(MPI_COMM_WORLD, &solver);
  HYPRE_SStructFACSetMaxLevels(solver, numLevels);
  HYPRE_SStructFACSetMaxIter(solver, 20);
  HYPRE_SStructFACSetTol(solver, 1.0e-06);
  HYPRE_SStructFACSetPLevels(solver, numLevels, _pLevel);
  HYPRE_SStructFACSetPRefinements(solver, numLevels, _refinementRatio);
  HYPRE_SStructFACSetRelChange(solver, 0);
  HYPRE_SStructFACSetRelaxType(solver, 2); // or 1
  HYPRE_SStructFACSetNumPreRelax(solver, n_pre);
  HYPRE_SStructFACSetNumPostRelax(solver, n_post);
  HYPRE_SStructFACSetCoarseSolverType(solver, 2);
  HYPRE_SStructFACSetLogging(solver, 1);
  HYPRE_SStructFACSetup2(solver, _facA, _b, _x);
      
  hypre_FacZeroCData(solver, _facA, _b, _x);

  hypre_EndTiming(time_index);
  hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  time_index = hypre_InitializeTiming("FAC Solve");
  hypre_BeginTiming(time_index);

  dbg0 << "----------------------------------------------------" << "\n";
  dbg0 << "calling FAC" << "\n";
  dbg0 << "----------------------------------------------------" << "\n";
  HYPRE_SStructFACSolve3(solver, _facA, _b, _x);

  hypre_EndTiming(time_index);
  hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  int numIterations = -1;
  HYPRE_SStructFACGetNumIterations(solver, &numIterations);
  _results.numIterations = numIterations;

  HYPRE_SStructFACGetFinalRelativeResidualNorm(solver,
                                               &_results.finalResNorm);
  HYPRE_SStructFACDestroy2(solver);


  /*-----------------------------------------------------------
   * Gather the solution vector
   *-----------------------------------------------------------*/
  dbg0 << "----------------------------------------------------" << "\n";
  dbg0 << "Gather the solution vector" << "\n";
  dbg0 << "----------------------------------------------------" << "\n";

  HYPRE_SStructVectorGather(_x);
} //end solve()

void
SolverFAC::printMatrix(const string& fileName /* = "solver" */)
{
  dbg << "SolverFAC::printMatrix() begin" << "\n";
  if (!_param->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);

  // TODO: implement facA printout in SolverFAC in addition to the
  // generic printMatrix()
  //  HYPRE_SStructMatrixPrint((fileName + ".fac").c_str(), _facA, 0);

  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
    /* Print CSR matrix in IJ format, base 1 for rows and cols */
    HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
  }
  dbg << "SolverFAC::printMatrix() end" << "\n";
}
