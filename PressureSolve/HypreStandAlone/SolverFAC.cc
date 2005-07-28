#include "SolverFAC.h"

#include "util.h"
#include "Level.h"
#include "Patch.h"

#include <string>

using namespace std;

void
SolverFAC::initialize(const Hierarchy& hier,
                      const HYPRE_SStructGrid& grid,
                      const HYPRE_SStructStencil& stencil,
                      const HYPRE_SStructGraph& graph)
{
  initializeData(hier, grid, graph);
  makeLinearSystem(hier, grid, stencil);
  assemble();
}

void
SolverFAC::initializeData(const Hierarchy& hier,
                          const HYPRE_SStructGrid& grid,
                          const HYPRE_SStructGraph& graph)
{
#if FAC
  /* Initialize arrays needed by Hypre FAC */
  const Counter numLevels = hier._levels.size();
  _pLevel          = hypre_TAlloc(int  , numLevels);    
  _refinementRatio = hypre_TAlloc(Index, numLevels);
  for (Counter level = 0; level < numLevels; level++) {
    _pLevel[level] = level;   // part ID of this level
    ToIndex(hier._levels[level]->_refRat, &_refinementRatio[level],
            _param->numDims);
  }
#endif
  
  _requiresPar =
    (((_solverID >= 20) && (_solverID <= 30)) ||
     ((_solverID >= 40) && (_solverID < 60)));

  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &_A);
  Print("Created empty SStructMatrix\n");
  /* If using AMG, set A's object type to ParCSR now */
  if (_requiresPar) {
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(_A);

  /* Initialize RHS vector b and solution vector x */
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  /* If AMG is used, set b and x type to ParCSR */
  if (_requiresPar) {
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_b);
  HYPRE_SStructVectorInitialize(_x);
}

void
SolverFAC::assemble(void)
{
  /* Assemble the matrix - a collective call */
  HYPRE_SStructMatrixAssemble(_A); 
  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if (_requiresPar) {
    HYPRE_SStructMatrixGetObject(_A, (void **) &_parA);
  }
  HYPRE_SStructVectorAssemble(_b);
  HYPRE_SStructVectorAssemble(_x);
 
  /* For BoomerAMG solver: set up the linear system (b,x) in ParCSR format */
  if (_requiresPar) {
    HYPRE_SStructVectorGetObject(_b, (void **) &_parB);
    HYPRE_SStructVectorGetObject(_x, (void **) &_parX);
  }
}

void
SolverFAC::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  Print("----------------------------------------------------\n");
  Print("FAC setup phase\n");
  Print("----------------------------------------------------\n");

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
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Gather the solution vector\n");
  Proc0Print("----------------------------------------------------\n");

  HYPRE_SStructVectorGather(_x);
} //end solve()

void
SolverFAC::printMatrix(const string& fileName /* = "solver" */)
{
  if (!_param->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);

  // TODO: implement facA printout in SolverFAC in addition to the
  // generic printMatrix()
  HYPRE_SStructMatrixPrint((fileName + ".fac").c_str(), _facA, 0);

  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
    /* Print CSR matrix in IJ format, base 1 for rows and cols */
    HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
  }
}
