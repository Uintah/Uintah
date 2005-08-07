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

#include "DebugStream.h"
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

int MYID;     // The same as this proc's myid, but global
DebugStream dbg("DEBUG",true);
DebugStream dbg0("DEBUG0",false);

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
  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars
  const Counter numDims   = param->numDims;
  const Counter numLevels = hier._levels.size();

  /* Create an empty grid in numDims dimensions with # parts = numLevels. */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);

  serializeProcsBegin();
  dbg << "makeGrid() begin" << "\n";
  /* Add the patches that this proc owns at all levels to grid */
  for (Counter level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    dbg
        << "Level " << level 
        << ", meshSize = " << lev->_meshSize[0]
        << ", resolution = " << lev->_resolution << "\n";
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      /* Add this patch to the grid */
      HYPRE_SStructGridSetExtents(grid, level,
                                  patch->_box.get(Left ).getData(),
                                  patch->_box.get(Right).getData());
      HYPRE_SStructGridSetVariables(grid, level, NUM_VARS, vars);
      dbg
          << "  Patch " << setw(2) << left << i
          << ", ID "    << setw(2) << left << patch->_patchID << " ";
      dbg << patch->_box;
      dbg << "\n";
    }
  }

  /*
    Assemble the grid; this is a collective call that synchronizes
    data from all processors. On exit from this function, the grid is
    ready.
  */
  //  dbg << "Before the end of makeGrid()" << "\n";
  serializeProcsEnd();

  HYPRE_SStructGridAssemble(grid);
  dbg0 << "" << "\n";
  dbg0 << "Assembled grid, "
       << "num parts " << hypre_SStructGridNParts(grid) << "\n";
    dbg0 << "\n";
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
  //  dbg << "stencilSize = %d   numDims = %d\n",stencilSize,numDims);
  stencil_offsets.resize(0,stencilSize);
  int entry;
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  entry = 0;
  stencil_offsets[entry].resize(0,numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    //    dbg << "Init entry = %d, dim = %d\n",entry,dim);
    stencil_offsets[entry][dim] = 0;
  }
  for (Counter dim = 0; dim < numDims; dim++) {
    for (int s = Left; s <= Right; s += 2) {
      entry++;
      stencil_offsets[entry].resize(0,numDims);
      //      dbg << "entry = %d, dim = %d\n",entry,dim);
      for (Counter d = 0; d < numDims; d++) {
        //        dbg << "d = %d  size = %d\n",d,stencil_offsets[entry].size());
        stencil_offsets[entry][d] = 0;
      }
      //      dbg << "Setting entry = %d, dim = %d\n",entry,dim);
      stencil_offsets[entry][dim] = s;
    }
  }
  
  /* Add stencil entries */
  dbg0 << "Stencil offsets:" << "\n";
  for (entry = 0; entry < stencilSize; entry++) {
    HYPRE_SStructStencilSetEntry(stencil, entry,
                                 stencil_offsets[entry].getData(), 0);
    dbg0 << "    entry = " << entry
         << ",  stencil_offsets = " << stencil_offsets[entry] << "\n";
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
  param->printSystem   = true;

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
    cerr << "\n\nError: unknown solver type" << "\n";
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
  if (MYID == 0) {
    dbg0.setActive(true);
  }
  dbg.setVerboseLevel(param->verboseLevel);
  dbg0.setVerboseLevel(param->verboseLevel);
  dbg.setLevel(param->verboseLevel);
  dbg0.setLevel(param->verboseLevel);

  serializeProcsBegin();
  const int numLevels = param->numLevels;
  const int numDims   = param->numDims;

  linePrint("=",70);
  dbg0 << argv[0]
       << ": Hypre solver interface test program for diffusion PDEs" << "\n";
  linePrint("=",70);
  dbg0 << "" << "\n";

  linePrint("-",50);
  dbg0 << "Initialize some stuff" << "\n";
  linePrint("-",50);

  if (myid == 0) {
    /* Read and check arguments, parameters */
    dbg << "Checking arguments and parameters ... ";
    if ((param->solverType == Param::FAC) &&
        ((numLevels < 2) || (numDims != 3))) {
      cerr << "\n\nFAC solver needs a 3D problem and at least 2 levels."
           << "\n";
      clean();
      exit(1);
    }
    dbg << "done" << "\n";

    dbg << "Checking # procs ... ";
    dbg << "numProcs = " << numProcs << ", done" << "\n";
    //    int correct = mypow(2,numDims);
    int correct = int(pow(2.0,numDims));
    if (numProcs != correct) {
      cerr << "\n\nError, hard coded to " << correct
           << " processors in " << numDims << "-D for now." << "\n";
      clean();
      exit(1);
    }

    dbg << "\n";
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
  linePrint("-",50);
  dbg0 << "Set up the grid (AMR levels, patches)" << "\n";
  linePrint("-",50);

  serializeProcsEnd();
  hier.make();
  makeGrid(param, hier, grid);             // Make Hypre grid from hier
  hier.printPatchBoundaries();

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Set up the stencils on all the patchs" << "\n";
  linePrint("-",50);
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
  linePrint("-",50);
  dbg0 << "Initialize the solver (Set up SStruct graph, matrix)" << "\n";
  linePrint("-",50);
  solver->initialize(hier, grid, stencil);

  /* Print total time for setting up the grid, stencil, graph, solver */
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
  dbg << "End timing" << "\n";

  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Print out the system and initial guess" << "\n";
  linePrint("-",50);
  solver->printMatrix("output_A");
  solver->printRHS("output_b");
  solver->printSolution("output_x0");

  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Solver setup phase" << "\n";
  linePrint("-",50);
  solver->setup();  // Depends only on A

  /*-----------------------------------------------------------
   * Solve the linear system A*x=b
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Solve the linear system A*x=b" << "\n";
  linePrint("-",50);
  solver->solve();  // Depends on A and b

  /*-----------------------------------------------------------
   * Print the solution and other info
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Print the solution vector" << "\n";
  linePrint("-",50);
  solver->printSolution("output_x1");
  dbg0 << "Iterations = " << solver->_results.numIterations << "\n";
  dbg0 << "Final Relative Residual Norm = "
       << solver->_results.finalResNorm << "\n";
  dbg0 << "" << "\n";

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  linePrint("-",50);
  dbg0 << "Finalize things" << "\n";
  linePrint("-",50);

  /* Destroy grid objects */
  dbg << "Destroying grid objects" << "\n";
  HYPRE_SStructGridDestroy(grid);
   
  /* Destroy stencil objects */
  dbg << "Destroying stencil objects" << "\n";
  HYPRE_SStructStencilDestroy(stencil);
   
  delete param;
  delete solver;
   
  clean();

  dbg << argv[0] << ": Going down successfully" << "\n";

  return 0;
} // end main()
