#include "Solver.h"

#include "util.h"
#include <string>
#include <map>

using namespace std;

void
Solver::initialize(const Hierarchy& hier,
                   const HYPRE_SStructGrid& grid,
                   const HYPRE_SStructGraph& graph)
{
  /* Initialize arrays needed by Hypre FAC */
  const Counter numLevels = hier._levels.size();
  _pLevel          = hypre_TAlloc(int  , numLevels);    
  _refinementRatio = hypre_TAlloc(Index, numLevels);
  for (Counter level = 0; level < numLevels; level++) {
    _pLevel[level] = level;   // part ID of this level
  }

  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &_A);
  Print("Created empty SStructMatrix\n");
  /* If using AMG, set A's object type to ParCSR now */
  if ( ((_param.solverID >= 20) && (_param.solverID <= 30)) ||
       ((_param.solverID >= 40) && (_param.solverID < 60)) ) {
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(_A);

  /* Initialize RHS vector b and solution vector x */
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  /* If AMG is used, set b and x type to ParCSR */
  if ( ((_param.solverID >= 20) && (_param.solverID <= 30)) ||
       ((_param.solverID >= 40) && (_param.solverID < 60)) ) {
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_b);
  HYPRE_SStructVectorInitialize(_x);
}

void
Solver::assemble(void)
{
  /* Assemble the matrix - a collective call */
  HYPRE_SStructMatrixAssemble(_A); 
  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if ( ((_param.solverID >= 20) && (_param.solverID <= 30)) ||
       ((_param.solverID >= 40) && (_param.solverID < 60)) ) {
    HYPRE_SStructMatrixGetObject(_A, (void **) &_parA);
  }

  hypre_ZeroAMRVectorData(_b, _pLevel, _refinementRatio);  // Implement ourselves?
  HYPRE_SStructVectorAssemble(_b);
  hypre_ZeroAMRVectorData(_x, _pLevel, _refinementRatio);
  HYPRE_SStructVectorAssemble(_x);  // See above
 
  /* For BoomerAMG solver: set up the linear system (b,x) in ParCSR format */
  if ( ((_param.solverID >= 20) && (_param.solverID <= 30)) ||
       ((_param.solverID >= 40) && (_param.solverID < 60)) ) {
    HYPRE_SStructVectorGetObject(_b, (void **) &_parB);
    HYPRE_SStructVectorGetObject(_x, (void **) &_parX);
  }
}

void
Solver::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  if (MYID == 0) {
    Print("----------------------------------------------------\n");
    Print("Solver setup phase\n");
    Print("----------------------------------------------------\n");
  }
  if (_param.solverID > 90) {
    /* FAC Solver. Prepare FAC operator hierarchy using Galerkin coarsening
       with black-box interpolation, on the original meshes */
    int time_fac_rap = hypre_InitializeTiming("fac rap");
    hypre_BeginTiming(time_fac_rap);
    hypre_AMR_RAP(_A, _refinementRatio, &_facA);
    hypre_ZeroAMRVectorData(_b, _pLevel, _refinementRatio);
    hypre_ZeroAMRVectorData(_x, _pLevel, _refinementRatio);
    hypre_EndTiming(time_fac_rap);
    hypre_PrintTiming("fac rap", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_fac_rap);
    hypre_ClearTiming();
  }
}
