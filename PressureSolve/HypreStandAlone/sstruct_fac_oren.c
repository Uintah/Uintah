#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "krylov.h"
#include "sstruct_mv.h"
#include "sstruct_ls.h"
 
#define DEBUG 0

#if DEBUG
#  include "sstruct_mv.h"
#endif

#define MAX_DIMS 3
typedef int Index[MAX_DIMS];

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/
 
void printIndex(Index a, int numDims) {
  /* Print numDims-dimensional index a */
  int d;
  printf("[");
  for (d = 0; d < numDims; d++) {
    printf("%d",a[d]);
    if (d < numDims-1) printf(",");
  }
  printf("]");
}

void faceBoxExtents(int numDims, Index ilower, Index iupper,
                    int dim, int side,
                    Index* faceLower, Index* faceUpper) {
  /* Compute face box extents of a numDims-dimensional patch whos extents
     are ilower,iupper. This is the face in the dim-dimension; side = -1
     means the left face, side = 1 the right face (so dim=1, side=-1 is the
     x-left face). Face extents are returned in faceLower, faceUpper. */
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
  printf("Face(dim = %c, side = %d) box extents: ",dim+'x',side);
  printIndex(*faceLower,numDims);
  printf(" to ");
  printIndex(*faceUpper,numDims);
  printf("\n");
}


void loopHypercube(int numDims, Index ilower, Index iupper,
                   Index step, Index* list) {
  /* 
     Prepare a list of all cell indices in the hypercube specified by
     ilower,iupper, as if we're looping over them (numDims nested loops)
     with step size step. The output list of indices is return in list.
     list has to be allocated outside this function.
  */
  int done = 0, currentDim = 0, count = 0, d;
  Index cellIndex;
  for (d = 0; d < numDims; d++) cellIndex[d] = ilower[d];
  while (!done) {
    printf("  count = %2d  cellIndex = ",count);
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

int
main( int argc, char *argv[] )
{
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/

  /* Counters, specific sizes of this problem */
  int num_procs, myid, level, entry, dim, d, cell, side;
  int solver_id = 30; // 99 = FAC
  int numDims = 2; // 2D problem
  int numLevels = 2; // Number of AMR levels
  int n = 8;  // Level 0 grid size in every direction

  /* Grid data structures */
  HYPRE_SStructGrid grid;
  int *plevels;
  Index faceLower, faceUpper;
  double *h; // meshsize
  double meshSize;

  /* Stencil data structures */
  HYPRE_SStructStencil stencil;
  int stencil_size;
  Index *stencil_offsets;

  /* Graph data structures */
  HYPRE_SStructGraph graph, fac_graph;

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
  int num_iterations, n_pre, n_post;
  double final_res_norm;
          
  /* Timers, debugging flags */
  int time_index, time_fac_rap;
  int print_system = 1;

  printf("got to here\n");

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  printf("num procs:  %d\n", num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  //  hypre_InitMemoryDebug(myid);

  printf("i am:  %d\n", myid);

  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  if( myid == 0 ) {
    /* Read and check arguments, parameters */
    if ((solver_id > 90) && (numLevels < 2)) {
      fprintf(stderr,"FAC solver needs at least two levels.");
      MPI_Finalize();
      exit(1);
    }

    printf("========================================================\n");
    printf("%s : FAC Hypre solver interface test program\n",argv[0]);
    printf("========================================================\n\n");
   
    printf("----------------------------------------------------\n");
    printf("Initialize some stuff\n");
    printf("----------------------------------------------------\n");
  }

  /*-------------------------------------------------------------------------
   * Set up grid (including the AMR levels and refinement ratios)
   *-------------------------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the grid\n");
  printf("----------------------------------------------------\n");
   
  if( num_procs != 4 ) {
    printf("Error, hard coded to 4 processors for now.\n");
    // Quit MPI
    MPI_Finalize();
    exit( 1 );
  }

  int numParts = 8;

  HYPRE_SStructGridCreate( MPI_COMM_WORLD, numDims, numParts, &grid );

  int partA = 'a' - 'a';
  int partB = 'b' - 'a';
  int partC = 'c' - 'a';
  int partD = 'd' - 'a';
  int partE = 'e' - 'a';
  int partF = 'f' - 'a';
  int partG = 'g' - 'a';

  int ilower[numParts][numDims];
  int iupper[numParts][numDims];

  HYPRE_SStructVariable vars[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};

  int part = myid;

  int procMap[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };

  // Large part
  ilower[part][0] = procMap[myid][0]*16;
  ilower[part][1] = procMap[myid][1]*16;
  iupper[part][0] = ilower[part][0] + 15;
  iupper[part][1] = ilower[part][0] + 15;

  //sleep(1);
  HYPRE_SStructGridSetExtents( grid, part, ilower[part], iupper[part] );
  HYPRE_SStructGridSetVariables(grid, part, 1, vars);
  
  // Small part
  ilower[part+4][0] = 16 + procMap[myid][0]*16;;
  ilower[part+4][1] = 16 + procMap[myid][0]*16;;
  iupper[part+4][0] = ilower[part+4][0] + 15;
  iupper[part+4][1] = ilower[part+4][0] + 15;
  HYPRE_SStructGridSetExtents( grid, part+4, ilower[part+4], iupper[part+4] );

  HYPRE_SStructGridSetVariables(grid, part+4, 1, vars);

  /* Assemble the grid */
  printf("%d: NUM PARTS %d\n", myid, hypre_SStructGridNParts(grid));

  HYPRE_SStructGridAssemble(grid);

  printf("%d: num parts %d\n", myid, hypre_SStructGridNParts(grid));

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the stencils on all the procs\n");
  printf("----------------------------------------------------\n");
     
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
  
  printf("Creating stencil ...\n");
  /* Create an empty stencil */
  HYPRE_SStructStencilCreate(numDims, stencil_size, &stencil);
  printf("Created stencil\n");
  
  /* Add stencil entries */
  printf("Stencil offsets:\n");
  for (entry = 0; entry < stencil_size; entry++) {
    printf("    entry = %d,  stencil_offsets = ",entry);
    printIndex(stencil_offsets[entry],numDims);
    printf("\n");
    HYPRE_SStructStencilSetEntry(stencil, entry,
                                 stencil_offsets[entry], 0);
  }

#if 0
  // Graph stuff
  
#endif
  

  // Quit MPI
  MPI_Finalize();

} // end main()

