/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface.
 * This is a stand-alone hypre interface that uses FAC / AMG solvers to solve
 * the pressure equation in implicit AMR-ICE.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

#include "utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "krylov.h"
#include "sstruct_mv.h"
#include "sstruct_ls.h"
 
#define DEBUG    1
#define MAX_DIMS 3

typedef int Index[MAX_DIMS];
int     MYID;  /* The same as myid, but global */

int mypow(int x, int y) {
  /*_____________________________________________________________________
    Function mypow:
    Compute x^y
    _____________________________________________________________________*/
  int i,result = 1;
  for (i = 1; i <= y; i++) result *= x;
  return result;
}

void Print(char *fmt, ...) {
  /*_____________________________________________________________________
    Function Print:
    Print an output line on the current processor. Useful to parse MPI output.
    _____________________________________________________________________*/
  int vb = 1; /* Verbose level */
  va_list ap;
  va_start(ap, fmt);
  if (vb) {
    printf("PROC %2d: ",MYID);
    vprintf(fmt, ap);
  }
  fflush(stdout);
  if (vb) {
    va_start(ap, fmt);
    //    if (log_file)
    //      vfprintf(log_file, fmt, ap);
    //    if (log_file)
    //      fflush(log_file);
  }
  va_end(ap);
}

void printIndex(Index a, int numDims) {
  /*_____________________________________________________________________
    Function printIndex:
    Print numDims-dimensional index a
    _____________________________________________________________________*/
  int d;
  printf("[");
  for (d = 0; d < numDims; d++) {
    printf("%d",a[d]);
    if (d < numDims-1) printf(",");
  }
  printf("]");
}

void faceExtents(int numDims, Index ilower, Index iupper,
                 int dim, int side,
                 Index* faceLower, Index* faceUpper) {
  /*_____________________________________________________________________
    Function faceExtents:
    Compute face box extents of a numDims-dimensional patch whos extents
    are ilower,iupper. This is the face in the dim-dimension; side = -1
    means the left face, side = 1 the right face (so dim=1, side=-1 is the
    x-left face). Face extents are returned in faceLower, faceUpper
    _____________________________________________________________________*/
  int d;
  for (d = 0; d < numDims; d++) {
    (*faceLower)[d] = ilower[d];
    (*faceUpper)[d] = iupper[d];
  }
  if (side < 0) {
    (*faceUpper)[dim] = (*faceLower)[dim];
  } else {
    (*faceLower)[dim] = (*faceUpper)[dim];
  }
  Print("Face(dim = %c, side = %d) box extents: ",dim+'x',side);
  printIndex(*faceLower,numDims);
  printf(" to ");
  printIndex(*faceUpper,numDims);
  printf("\n");
}

void loopHypercube(int numDims, Index ilower, Index iupper,
                   Index step, Index* list) {
  /*_____________________________________________________________________
    Function loopHypercube:
    Prepare a list of all cell indices in the hypercube specified by
    ilower,iupper, as if we're looping over them (numDims nested loops)
    with step size step. The output list of indices is return in list.
    list has to be allocated outside this function.
    _____________________________________________________________________*/
  int done = 0, currentDim = 0, count = 0, d;
  Index cellIndex;
  for (d = 0; d < numDims; d++) cellIndex[d] = ilower[d];
  while (!done) {
    Print("  count = %2d  cellIndex = ",count);
    printIndex(cellIndex,numDims);
    for (d = 0; d < numDims; d++) list[count][d] = cellIndex[d];
    count++;
    printf("\n");
    cellIndex[currentDim]++;
    if (cellIndex[currentDim] > iupper[currentDim]) {
      while (cellIndex[currentDim] > iupper[currentDim]) {
        cellIndex[currentDim] = ilower[currentDim];
        currentDim++;
        if (currentDim >= numDims) {
          done = 1;
          break;
        }
        cellIndex[currentDim]++;
      }
    }
    if (done) break;
    currentDim = 0;
  }
}

int clean(void) {
  /*_____________________________________________________________________
    Function clean:
    Exit MPI, debug modes. Call before each exit() call and in the end
    of the program.
    _____________________________________________________________________*/
#if DEBUG
  hypre_FinalizeMemoryDebug();
#endif
  MPI_Finalize();    // Quit MPI
}

int main(int argc, char *argv[]) {
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/
  /* Counters, specific sizes of this problem */
  int numProcs, myid, level, entry, dim, d, cell, side;
  int solver_id = 30; // solver ID. 30 = AMG, 99 = FAC
  int numDims   = 2;  // 2D problem
  int numLevels = 2;  // Number of AMR levels
  int n         = 8;  // Level 0 grid size in every direction
  
  /* Grid data structures */
  HYPRE_SStructGrid grid;
  int               *levelID;
  Index             *ilower,*iupper,*refinementRatio;
  Index             faceLower, faceUpper;
  double            *h; // meshsize at all levels
  double            meshSize;
  
  /* Stencil data structures. We use the same stencil all all levels and all parts. */
  HYPRE_SStructStencil stencil;
  int                  stencil_size;
  Index                *stencil_offsets;
  
  /* Graph data structures */
  HYPRE_SStructGraph   graph, fac_graph;
  
  /* For matrix setup */
  //  Index index, to_index;
  double *values; //, *box_values;

  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  HYPRE_SStructMatrix   A, fac_A;
  HYPRE_SStructVector   b;
  HYPRE_SStructVector   x;
  HYPRE_SStructSolver   solver;
  HYPRE_ParCSRMatrix    par_A;
  HYPRE_ParVector       par_b;
  HYPRE_ParVector       par_x;
  HYPRE_Solver          par_solver;
  int                   num_iterations, n_pre, n_post;
  double                final_res_norm;
  
  /* Timers, debugging flags */
  int                   time_index, time_fac_rap;
  int                   print_system = 1;
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MYID = myid;
#if DEBUG
  hypre_InitMemoryDebug(myid);
#endif
  Print("<<<############# I am proc %d #############>>>\n", myid);
  
  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("========================================================\n");
    Print("%s : FAC Hypre solver interface test program\n",argv[0]);
    Print("========================================================\n");
    Print("\n");
    Print("----------------------------------------------------\n");
    Print("Initialize some stuff\n");
    Print("----------------------------------------------------\n");

    /* Read and check arguments, parameters */
    Print("Checking arguments and parameters ... ");
    if ((solver_id > 90) && (numLevels < 2)) {
      fprintf(stderr,"FAC solver needs at least two levels.");
      clean();
      exit(1);
    }
    printf("done\n");

    Print("Checking # procs ... ");
    int correct = mypow(2,numDims);
    if (numProcs != correct) {
      Print("\n\nError, hard coded to %d processors in %d-D for now.\n",correct,numDims);
      clean();
      exit(1);
    }
    printf("numProcs = %d, done\n",numProcs);

    Print("\n");
  }
  time_index = hypre_InitializeTiming("SStruct Interface");
  hypre_BeginTiming(time_index);

  /*-------------------------------------------------------------------------
   Set up grid (AMR levels, patches)
   Geometry:
   2D rectangular domain. 
   Finite volume discretization, cell centered nodes.
   One level.
   Level 0 mesh is uniform meshsize h = 1.0 in x and y. Cell numbering is
   (0,0) (bottom left corner) to (n-1,n-1) (upper right corner).
   Level 1 is of meshsize h/2 and extends over the central half of the domain.
   We have 4 processors, so each processor gets 2 patches: a quarter of level
   0 and a quarter of level 1. Each processor gets the patches covering a
   quadrant of the physical domain.
   *-------------------------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the grid (AMR levels, patches)\n");
    Print("----------------------------------------------------\n");
  }

  /* Create an empty grid in numDims dimensions with numParts parts (=patches) */
  int numParts    = numLevels*numProcs;
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numParts, &grid);
  HYPRE_SStructVariable vars[1] = {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars

  /* Initialize arrays holding level and parts info */
  levelID         = hypre_TAlloc(int, numLevels);    
  refinementRatio = hypre_TAlloc(Index, numParts);
  ilower          = hypre_CTAlloc(Index, numParts);
  iupper          = hypre_CTAlloc(Index, numParts);
  h               = hypre_CTAlloc(double, numParts);

  int part = myid;
  int procMap[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} }; // Works for 2D; write general gray code

  /* Initialize the parts at all levels that this proc owns */
  Print("Number of levels = %d\n",numLevels);
  for (level = 0; level < numLevels; level++) {
    Print("  Initializing level %d\n",level);
    levelID[level] = level;   /* Level ID */

    /* Refinement ratio w.r.t. parent level. Assumed to be constant
       (1:2) in all dimensions and all levels for now. */
    if (level == 0) {
      /* Dummy value */
      for (dim = 0; dim < numDims; dim++) {
        refinementRatio[level][dim] = 0;
      }
    } else {
      for (dim = 0; dim < numDims; dim++) {
        refinementRatio[level][dim] = 2;
      }
    }
    
    /* Compute meshsize, assumed the same in all directions at each level */
    if (level == 0) {
      h[level] = 1.0/n;   // Size of domain divided by # of gridpoints
    } else {
      h[level] = h[level-1] / refinementRatio[level][0]; // ref. ratio constant for all dims
    }
    
    /* Mesh box extents (lower left corner, upper right corner) */
    switch (level) {
    case 0:
      /* Level 0 extends over the entire domain. */
      {
        part = myid;   // Part number on level 0 for this proc
        for (dim = 0; dim < numDims; dim++) {
          ilower[level][dim] = procMap[myid][dim] * n;
          iupper[level][dim] = ilower[level][dim] + n - 1;
        }
        /* Add this patch to the grid */
        HYPRE_SStructGridSetExtents( grid, part, ilower[part], iupper[part] );
        HYPRE_SStructGridSetVariables(grid, part, 1, vars);
        Print("  Part %d Extents = ",part);
        printIndex(ilower[level],numDims);
        printf(" to ");
        printIndex(iupper[level],numDims);
        printf("\n");
        fflush(stdout);
        break;
      }
    case 1:
      /* Level 1 extends over central square of physical size 1/2x1/2. */
      {
        part = myid + numParts/2;   // Part number on level 0 for this proc
        for (dim = 0; dim < numDims; dim++) {
          ilower[level][dim] = n + procMap[myid][dim] * n;
          iupper[level][dim] = ilower[level][dim] + n - 1;
        }
        /* Add this patch to the grid */
        HYPRE_SStructGridSetExtents( grid, part, ilower[part], iupper[part] );
        HYPRE_SStructGridSetVariables(grid, part, 1, vars);
        Print("  Part %d Extents = ",part);
        printIndex(ilower[level],numDims);
        printf(" to ");
        printIndex(iupper[level],numDims);
        printf("\n");
        fflush(stdout);

        break;
      }
    default: {
      Print("Unknown level - cannot set grid extents\n");
      clean();
      exit(1);
    }
    }
  }
  
  /* Assemble the grid; this is a collective call that synchronizes
     data from all processors. */
  HYPRE_SStructGridAssemble(grid);
  Print("\n");
  Print("Assembled grid num parts %d\n", hypre_SStructGridNParts(grid));
  //  MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  if (myid == 0) {
    Print("\n");
  }

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the stencils on all the parts\n");
    Print("----------------------------------------------------\n");
  }
  /* 
     The following code is for general dimension.
     We use a 5-point FD discretization to L = -Laplacian in this example.
     stencil_offsets specifies the non-zero pattern of the stencil;
     Its entries are also defined here and assumed to be constant over the
     structured mesh. If not, define it later during matrix setup.
  */
  stencil_size = 2*numDims+1;
  stencil_offsets = hypre_CTAlloc(Index, stencil_size);
  entry = 0;
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  for (d = 0; d < numDims; d++) {
    stencil_offsets[entry][d] = 0;
  }
  entry++;
  for (dim = 0; dim < numDims; dim++) {
    for (side = -1; side <= 1; side += 2) {
      for (d = 0; d < numDims; d++) {
        stencil_offsets[entry][d] =  0;
      }
      stencil_offsets[entry][dim] = side;
      entry++;
    }
  }
  
  Print("Creating stencil ... ");
  /* Create an empty stencil */
  HYPRE_SStructStencilCreate(numDims, stencil_size, &stencil);
  printf("done\n");
  
  /* Add stencil entries */
  if (myid == 0) {
    Print("Stencil offsets:\n");
  }
  for (entry = 0; entry < stencil_size; entry++) {
    HYPRE_SStructStencilSetEntry(stencil, entry,
                                 stencil_offsets[entry], 0);
    if (myid == 0) {
      Print("    entry = %d,  stencil_offsets = ",entry);
      printIndex(stencil_offsets[entry],numDims);
      printf("\n");
      fflush(stdout);
    }
  }

  /*-----------------------------------------------------------
   * Set up the graph
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the graph\n");
    Print("----------------------------------------------------\n");
  }

  /* Create an empty graph */
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
  /* If using AMG, set graph's object type to ParCSR now */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
  }

  // Graph stuff to be added

  /* Assemble the graph */
  HYPRE_SStructGraphAssemble(graph);
  Print("Assembled graph, nUVentries = %d\n",hypre_SStructGraphNUVEntries(graph));
  
  /*-----------------------------------------------------------
   * Set up the SStruct matrix
   *-----------------------------------------------------------*/
  /*
    A = the original composite grid operator.
    Residual norms in any solver are measured as ||b-A*x||. 
    If FAC solver is used, it creates a new matrix fac_A from A, and uses
    Galerkin coarsening to replace the equations in a coarse part underlying
    a fine part.
  */
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the SStruct matrix\n");
    Print("----------------------------------------------------\n");
  }
  
  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  /* If using AMG, set A's object type to ParCSR now */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(A);
  
  // Add here interior equations of each part to A
  
  /* 
     Zero out all the connections from fine point stencils to outside the
     fine patch. These are replaced by the graph connections between the fine
     part and its parent coarse part.
  */
  for (level = numLevels-1; level > 0; level--) {
    hypre_FacZeroCFSten(hypre_SStructMatrixPMatrix(A, level),
                        hypre_SStructMatrixPMatrix(A, level-1),
                        grid,
                        level,
                        refinementRatio[level]);
    hypre_FacZeroFCSten(hypre_SStructMatrixPMatrix(A, level),
                        grid,
                        level);
    hypre_ZeroAMRMatrixData(A, level-1, refinementRatio[level]);
  }
  
  /* Assemble the matrix */
  HYPRE_SStructMatrixAssemble(A);
  
  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
  }
  
  /*-----------------------------------------------------------
   * Set up the RHS (b) and LHS (x) vectors
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the RHS (b), LHS (x) vectors\n");
    Print("----------------------------------------------------\n");
  }
  
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
  /* If AMG is used, set b and x type to ParCSR */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(b);
  HYPRE_SStructVectorInitialize(x);

  /* Initialize b at all levels */
  printf("Adding structured equations to b and x\n");
  for (level = 0; level < numLevels; level++) {
    printf("At level = %d\n",level);
    /* May later need to loop over patches at this level here. */
    /* b is defined only over interior structured equations.
       Graph not involved. B.C. are assumed to be eliminated.
    */
    
    /* Init values to vector of size = number of cells in the mesh */
    int numCells = 1;
    for (dim = 0; dim < numDims; dim++) {
      numCells *= (iupper[level][dim] - ilower[level][dim] + 1);
    }
    values = hypre_TAlloc(double, numCells);
    printf("numCells = %d\n",numCells);

    printf("-- Initializing b\n");
    for (cell = 0; cell < numCells; cell++) {
      values[cell] = 1.0;   /* RHS value at cell */
    } // for cell
    HYPRE_SStructVectorSetBoxValues(b, level, ilower[level], iupper[level],
                                    0, values);

    printf("-- Initializing x\n");
    for (cell = 0; cell < numCells; cell++) {
      values[cell] = 1.0;   /* Initial guess for LHS - value at cell */
    } // for cell
    HYPRE_SStructVectorSetBoxValues(x, level, ilower[level], iupper[level],
                                    0, values);
    
    hypre_TFree(values);
    
  } // for level

  hypre_ZeroAMRVectorData(b, levelID, refinementRatio);  // Do we need this?
  HYPRE_SStructVectorAssemble(b);
  hypre_ZeroAMRVectorData(x, levelID, refinementRatio);
  HYPRE_SStructVectorAssemble(x);  // See above
 
  /* For BoomerAMG solver: set up the linear system (b,x) in ParCSR format */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructVectorGetObject(b, (void **) &par_b);
    HYPRE_SStructVectorGetObject(x, (void **) &par_x);
  }
  
  /* Print total time for setting up the linear system */
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
  
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Solver setup phase\n");
    Print("----------------------------------------------------\n");
  }
  if (solver_id > 90) {
    /* FAC Solver. Prepare FAC operator hierarchy using Galerkin coarsening
       with black-box interpolation, on the original meshes */
    time_fac_rap = hypre_InitializeTiming("fac rap");
    hypre_BeginTiming(time_fac_rap);
    hypre_AMR_RAP(A, refinementRatio, &fac_A);
    hypre_ZeroAMRVectorData(b, levelID, refinementRatio);
    hypre_ZeroAMRVectorData(x, levelID, refinementRatio);
    hypre_EndTiming(time_fac_rap);
    hypre_PrintTiming("fac rap", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_fac_rap);
    hypre_ClearTiming();
  }

  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Print out the system and initial guess\n");
    Print("----------------------------------------------------\n");
  }
  if (print_system) {
    HYPRE_SStructMatrixPrint("sstruct.out.A", A, 0);
    if (solver_id > 90) {
      HYPRE_SStructMatrixPrint("sstruct.out.facA", fac_A, 0);
    }
    if (solver_id == 30) {
      HYPRE_ParCSRMatrixPrint(par_A, "sstruct.out.parA");
      /* Print CSR matrix in IJ format, base 1 for rows and cols */
      HYPRE_ParCSRMatrixPrintIJ(par_A, 1, 1, "sstruct.out.ijA");
    }
    HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
    HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
  }

  /*-----------------------------------------------------------
   * Solve the linear system A*x=b
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Solve the linear system A*x=b\n");
    Print("----------------------------------------------------\n");
  }

  /*-------------- FAC Solver -----------------*/
  if (solver_id > 90) {
      n_pre  = refinementRatio[numLevels-1][0]-1;
      n_post = refinementRatio[numLevels-1][0]-1;

      /* n_pre+= n_post;*/
      /* n_post= 0;*/

      time_index = hypre_InitializeTiming("FAC Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructFACCreate(MPI_COMM_WORLD, &solver);
      HYPRE_SStructFACSetMaxLevels(solver, numLevels);
      HYPRE_SStructFACSetMaxIter(solver, 20);
      HYPRE_SStructFACSetTol(solver, 1.0e-06);
      HYPRE_SStructFACSetPLevels(solver, numLevels, levelID);
      HYPRE_SStructFACSetPRefinements(solver, numLevels, refinementRatio);
      HYPRE_SStructFACSetRelChange(solver, 0);
      if (solver_id > 90)
      {
         HYPRE_SStructFACSetRelaxType(solver, 2);
      }
      else
      {
         HYPRE_SStructFACSetRelaxType(solver, 1);
      }
      HYPRE_SStructFACSetNumPreRelax(solver, n_pre);
      HYPRE_SStructFACSetNumPostRelax(solver, n_post);
      HYPRE_SStructFACSetCoarseSolverType(solver, 2);
      HYPRE_SStructFACSetLogging(solver, 1);
      HYPRE_SStructFACSetup2(solver, fac_A, b, x);
      
      hypre_FacZeroCData(solver, fac_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FAC Solve");
      hypre_BeginTiming(time_index);

      if (solver_id > 90)
      {
         HYPRE_SStructFACSolve3(solver, fac_A, b, x);
      }
      else
      {
         HYPRE_SStructFACSolve3(solver, fac_A, b, x);
      }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructFACGetNumIterations(solver, &num_iterations);
      HYPRE_SStructFACGetFinalRelativeResidualNorm(
                                           solver, &final_res_norm);
      HYPRE_SStructFACDestroy2(solver);
   }

  /*-------------- AMG Solver -----------------*/
   if (solver_id == 30)
   {
      time_index = hypre_InitializeTiming("AMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&par_solver);
      HYPRE_BoomerAMGSetCoarsenType(par_solver, 6);
      HYPRE_BoomerAMGSetStrongThreshold(par_solver, 0.);
      HYPRE_BoomerAMGSetTruncFactor(par_solver, 0.3);
      /*HYPRE_BoomerAMGSetMaxLevels(par_solver, 4);*/
      HYPRE_BoomerAMGSetTol(par_solver, 1.0e-06);
      HYPRE_BoomerAMGSetPrintLevel(par_solver, 1);
      HYPRE_BoomerAMGSetPrintFileName(par_solver, "sstruct.out.log");
      HYPRE_BoomerAMGSetMaxIter(par_solver, 200);
      HYPRE_BoomerAMGSetup(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BoomerAMGGetNumIterations(par_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(par_solver,
                                                 &final_res_norm);

      HYPRE_BoomerAMGDestroy(par_solver);
   }

  /*-----------------------------------------------------------
   * Gather the solution vector
   *-----------------------------------------------------------*/
   if (myid == 0) {
     Print("----------------------------------------------------\n");
     Print("Gather the solution vector\n");
     Print("----------------------------------------------------\n");
   }

   HYPRE_SStructVectorGather(x);
   if (print_system) {
     HYPRE_SStructVectorPrint("sstruct.out.x1", x, 0);
   }
   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/
   if (myid == 0) {
     Print("----------------------------------------------------\n");
     Print("Print the solution vector\n");
     Print("----------------------------------------------------\n");
     Print("Iterations = %d\n", num_iterations);
     Print("Final Relative Residual Norm = %e\n", final_res_norm);
     Print("\n");
   }

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
   if (myid == 0) {
     Print("----------------------------------------------------\n");
     Print("Finalize things\n");
     Print("----------------------------------------------------\n");
   }

   /* Destroy grid objects */
   HYPRE_SStructGridDestroy(grid);
   hypre_TFree(levelID);
   hypre_TFree(refinementRatio);
   hypre_TFree(ilower);
   hypre_TFree(iupper);
   hypre_TFree(h);
   
   /* Destroy stencil objects */
   HYPRE_SStructStencilDestroy(stencil);
   hypre_TFree(stencil_offsets);
   
   /* Destroy graph objects */
   if (solver_id > 90) {
     fac_graph = hypre_SStructMatrixGraph(fac_A);
     HYPRE_SStructGraphDestroy(fac_graph);
   }
   HYPRE_SStructGraphDestroy(graph);
   
   /* Destroy matrix, RHS, solution objects */
   if (solver_id > 90) {
     HYPRE_SStructMatrixDestroy(fac_A);
   }
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   
   clean();
   return 0;
} // end main()
