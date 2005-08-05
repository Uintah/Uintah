/*-------------------------------------------------------------------------
 * File: mydriver.cc
 *
 * Test driver for semi-structured matrix interface.
 * This is a stand-alone hypre interface that uses FAC / AMG solvers to solve
 * the pressure equation in implicit AMR-ICE.
 *
 * Revision history:
 * 19-JUL-2005   Dav & Oren   Works&does nothing for 4 procs, doesn't crash.
 *-------------------------------------------------------------------------*/

/*================== Library includes ==================*/

#include "util.h"
#include "Hierarchy.h"

#include "TestLinear.h"

#include "Solver.h"
#include "SolverAMG.h"
#include "SolverFAC.h"

// Hypre includes
#include <HYPRE_sstruct_ls.h>
#include <utilities.h>
#include <krylov.h>
#include <sstruct_mv.h>
#include <sstruct_ls.h>

using namespace std;

/*================== Global variables ==================*/

int     MYID;     /* The same as this proc's myid, but global */

void
makeGrid(const Param* param,
         const Hierarchy& hier,
         HYPRE_SStructGrid& grid)
  /*_____________________________________________________________________
    Function makeGrid:
    Create an empty Hypre grid object "grid" from our hierarchy hier,
    and add all patches from this proc to it.
    _____________________________________________________________________*/
{
  Print("makeGrid() begin\n");
  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars
  const Counter numDims   = param->numDims;
  const Counter numLevels = hier._levels.size();

  /* Create an empty grid in numDims dimensions with # parts = numLevels. */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);

  serializeProcsBegin();
  /* Add the patches that this proc owns at all levels to grid */
  for (Counter level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    Print("Level %d, meshSize = %lf, resolution = ",
          level,lev->_meshSize[0]);
    cout << lev->_resolution << "\n";
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      /* Add this patch to the grid */
      HYPRE_SStructGridSetExtents(grid, level,
                                  patch->_box.get(Left ).getData(),
                                  patch->_box.get(Right).getData());
      HYPRE_SStructGridSetVariables(grid, level, NUM_VARS, vars);
      Print("  Patch %2d, ID %2d ",i,patch->_patchID);
      cout << patch->_box;
      PrintNP("\n");
    }
  }

  /*
    Assemble the grid; this is a collective call that synchronizes
    data from all processors. On exit from this function, the grid is
    ready.
  */
  //  Print("Before the end of makeGrid()\n");
  serializeProcsEnd();
  HYPRE_SStructGridAssemble(grid);
  if (MYID == 0) {
    Print("\n");
    Print("Assembled grid, num parts %d\n", hypre_SStructGridNParts(grid));
    Print("\n");
  }
}

void
makeStencil(const Param* param,
            const Hierarchy& hier,
            HYPRE_SStructStencil& stencil)
  /*_____________________________________________________________________
    Function makeStencil:
    Initialize the Hypre stencil with a 5-point stencil (in d dimensions).
    Create Hypre stencil object "stencil" on output.
    _____________________________________________________________________*/
{
  const Counter numDims   = param->numDims;
  Counter               stencilSize = 2*numDims+1;
  Vector< Vector<int> > stencil_offsets;

  /* Create an empty stencil */
  HYPRE_SStructStencilCreate(numDims, stencilSize, &stencil);
  
  serializeProcsBegin();
  /*
    The following code is for general dimension.
    We use a 5-point FD discretization to L = -Laplacian in this example.
    stencil_offsets specifies the non-zero pattern of the stencil;
    Its entries are also defined here and assumed to be constant over the
    structured mesh. If not, define it later during matrix setup.
  */
  //  Print("stencilSize = %d   numDims = %d\n",stencilSize,numDims);
  stencil_offsets.resize(0,stencilSize);
  int entry;
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  entry = 0;
  stencil_offsets[entry].resize(0,numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    //    Print("Init entry = %d, dim = %d\n",entry,dim);
    stencil_offsets[entry][dim] = 0;
  }
  for (Counter dim = 0; dim < numDims; dim++) {
    for (int s = Left; s <= Right; s += 2) {
      entry++;
      stencil_offsets[entry].resize(0,numDims);
      //      Print("entry = %d, dim = %d\n",entry,dim);
      for (Counter d = 0; d < numDims; d++) {
        //        Print("d = %d  size = %d\n",d,stencil_offsets[entry].size());
        stencil_offsets[entry][d] = 0;
      }
      //      Print("Setting entry = %d, dim = %d\n",entry,dim);
      stencil_offsets[entry][dim] = s;
    }
  }
  
  /* Add stencil entries */
  Proc0Print("Stencil offsets:\n");
  for (entry = 0; entry < stencilSize; entry++) {
    HYPRE_SStructStencilSetEntry(stencil, entry,
                                 stencil_offsets[entry].getData(), 0);
    if (MYID == 0) {
      Print("    entry = %d,  stencil_offsets = ",entry);
      cout << stencil_offsets[entry] << "\n";
    }
  }
  serializeProcsEnd();
}

int
main(int argc, char *argv[]) {
  /*-----------------------------------------------------------
   * Parameter initialization
   *-----------------------------------------------------------*/
  /* Set test cast parameters */
  Param*                param;
  param = new TestLinear(2,8); // numDims, baseResolution
  param->solverType    = Param::AMG; // Hypre solver
  param->numLevels     = 2;          // # AMR levels
  param->printSystem   = false; //true;

  /* Grid hierarchy & stencil objects */
  Hierarchy             hier(param);
  HYPRE_SStructGrid     grid;
  HYPRE_SStructStencil  stencil;     // Same stencil at all levels & patches

  /* Set up Solver object */
  Solver*               solver = 0;      // Solver data structure
  switch (param->solverType) {
  case Param::AMG:
    solver = new SolverAMG(param);
    break;
  case Param::FAC:
    solver = new SolverFAC(param);
    break;
  default:
    fprintf(stderr,"\n\nError: unknown solver type\n");
    clean();
    exit(1);
  }
  
  /*-----------------------------------------------------------
   * Initialize some stuff, check arguments
   *-----------------------------------------------------------*/
  /* Initialize MPI */
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  param->numProcs = numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MYID = myid;
#if DRIVER_DEBUG
  hypre_InitMemoryDebug(myid);
#endif
  const int numLevels = param->numLevels;
  const int numDims   = param->numDims;

  Proc0Print("========================================================\n");
  Proc0Print("%s : FAC Hypre solver interface test program\n",argv[0]);
  Proc0Print("========================================================\n");
  Proc0Print("\n");
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Initialize some stuff\n");
  Proc0Print("----------------------------------------------------\n");

  if (myid == 0) {
    /* Read and check arguments, parameters */
    Print("Checking arguments and parameters ... ");
    if ((param->solverType == Param::FAC) &&
        ((numLevels < 2) || (numDims != 3))) {
      PrintNP("FAC solver needs a 3D problem and at least 2 levels.");
      clean();
      exit(1);
    }
    PrintNP("done\n");

    Print("Checking # procs ... ");
    //    int correct = mypow(2,numDims);
    int correct = int(pow(2.0,numDims));
    if (numProcs != correct) {
      Print("\n\nError, hard coded to %d processors in %d-D for now.\n",
            correct,numDims);
      clean();
      exit(1);
    }
    PrintNP("numProcs = %d, done\n",numProcs);

    Print("\n");
  }

  int time_index = hypre_InitializeTiming("SStruct Interface");
  hypre_BeginTiming(time_index);

  /*----------------------------------------------------------------------
    Set up grid (AMR levels, patches)
    Geometry:
    2D rectangular domain. 
    Finite volume discretization, cell centered nodes.
    One level.
    Level 0 mesh is uniform meshsize h = 1.0 in x and y. Cell numbering is
    (0,0) (bottom left corner) to (n-1,n-1) (upper right corner).
    Level 1 is meshsize h/2 and extends over the central half of the domain.
    We have 4 processors, so each proc gets 2 patches: a quarter of level
    0 and a quarter of level 1. Each processor gets the patches covering a
    quadrant of the physical domain.
    *----------------------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the grid (AMR levels, patches)\n");
  Proc0Print("----------------------------------------------------\n");

  hier.make();
  makeGrid(param, hier, grid);             // Make Hypre grid from hier
  hier.printPatchBoundaries();

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the stencils on all the patchs\n");
  Proc0Print("----------------------------------------------------\n");
  makeStencil(param, hier, stencil);

  /*-----------------------------------------------------------
   * Set up the SStruct matrix
   *-----------------------------------------------------------*/
  /*
    A = the original composite grid operator.
    Residual norms in any solver are measured as ||b-A*x||. 
    If FAC solver is used, it creates a new matrix fac_A from A, and uses
    Galerkin coarsening to replace the equations in a coarse patch underlying
    a fine patch.
  */
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the SStruct matrix\n");
  Proc0Print("----------------------------------------------------\n");
  solver->initialize(hier, grid, stencil);

  /* Print total time for setting up the grid, stencil, graph, solver */
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
  Print("End timing\n");

  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Print out the system and initial guess\n");
  Proc0Print("----------------------------------------------------\n"); 
  solver->printMatrix("output_A");
  solver->printRHS("output_b");
  solver->printSolution("output_x0");

  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Solver setup phase\n");
  Proc0Print("----------------------------------------------------\n");
  solver->setup();  // Depends only on A

  /*-----------------------------------------------------------
   * Solve the linear system A*x=b
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Solve the linear system A*x=b\n");
  Proc0Print("----------------------------------------------------\n");
  solver->solve();  // Depends on A and b

  /*-----------------------------------------------------------
   * Print the solution and other info
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Print the solution vector\n");
  Proc0Print("----------------------------------------------------\n");
  solver->printSolution("output_x1");
  Proc0Print("Iterations = %d\n", solver->_results.numIterations);
  Proc0Print("Final Relative Residual Norm = %e\n",
             solver->_results.finalResNorm);
  Proc0Print("\n");

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Finalize things\n");
  Proc0Print("----------------------------------------------------\n");

  /* Destroy grid objects */
  Print("Destroying grid objects\n");
  HYPRE_SStructGridDestroy(grid);
   
  /* Destroy stencil objects */
  Print("Destroying stencil objects\n");
  HYPRE_SStructStencilDestroy(stencil);
   
  delete param;
  delete solver;
   
  clean();

  Print("%s: Going down successfully\n",argv[0]);

  return 0;
} // end main()
