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
main(int argc, char *argv[] )
{
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/
  printf("========================================================\n");
  printf("%s : FAC Hypre solver interface test program\n",argv[0]);
  printf("========================================================\n\n");
   
  /* Counters, specific sizes of this problem */
  int num_procs, myid, level, entry, dim, d, cell, side;
  int solver_id = 30; // 99 = FAC
  int numDims = 2; // 2D problem
  int numLevels = 2; // Number of AMR levels
  int n = 8;  // Level 0 grid size in every direction

  /* Grid data structures */
  HYPRE_SStructGrid grid;
  int *plevels;
  Index *ilower,*iupper,*prefinements;
  Index faceLower, faceUpper;
  double *h; // meshsize
  double meshSize;

  /* Stencil data structures */
  HYPRE_SStructStencil *stencils;
  int *stencil_sizes;
  Index **stencil_offsets;
  double **stencil_values;

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

  /* Read and check arguments, parameters */
  if ((solver_id > 90) && (numLevels < 2)) {
    fprintf(stderr,"FAC solver needs at least two levels.");
    exit(1);
  }

  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Initialize some stuff\n");
  printf("----------------------------------------------------\n");
   
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  hypre_InitMemoryDebug(myid);

  /*-------------------------------------------------------------------------
   * Set up grid (including the AMR levels and refinement ratios)
   *-------------------------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the grid\n");
  printf("----------------------------------------------------\n");
   
  time_index = hypre_InitializeTiming("SStruct Interface");
  hypre_BeginTiming(time_index);

  /* Geometry:
     2D rectangular domain. 
     Finite volume discretization, cell centered nodes.
     One level.
     Level 0 mesh is uniform meshsize h = 1.0 in x and y. Cell numbering is
     (1,1) (bottom left corner) to (n,n) (upper right corner).
  */
  
  /* Create empty grid of numDim dimensions */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);
  printf("Created grid in %d-D\n",numDims);

  /* Add level meshes to grid */
  plevels      = hypre_TAlloc(int, numLevels);
  prefinements = hypre_TAlloc(Index, numLevels);
  ilower       = hypre_CTAlloc(Index, numLevels);
  iupper       = hypre_CTAlloc(Index, numLevels);
  h            = hypre_CTAlloc(double, numLevels);
  
  printf("Number of levels = %d\n",numLevels);
  for (level = 0; level < numLevels; level++) {
    printf("level = %d\n",level);
    plevels[level] = level;   /* Level ID */

    /* Refinement ratio w.r.t. parent level. Assumed to be constant
       in all dimensions for now. */
    if (level == 0) {
      /* Dummy value */
      for (dim = 0; dim < numDims; dim++) {
        prefinements[level][dim] = 1;
      }
    } else {
      for (dim = 0; dim < numDims; dim++) {
        prefinements[level][dim] = 2;
      }
    }
    
    /* Compute meshsize */
    if (level == 0) {
      h[level] = 1.0/n;   /* Size of domain / no. of points */
    } else {
      h[level] = h[level-1] / prefinements[level][0]; /* See prefinement
                                                         comment above */
    }
    
    /* We use one type of variables at each cell: cell-centered nodes */
    HYPRE_SStructVariable vars[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
    
    /* Mesh box extents (lower left corner, upper right corner) */
    switch (level) {
    case 0: /* Level 0 extends over the entire domain (1x1).
               Indices: (1,1) to (8,8) */
      {
        for (dim = 0; dim < numDims; dim++) {
          ilower[level][dim] = 1;
          iupper[level][dim] = 8;
        }
        break;
      }
    case 1: /* Level 1 extends over central square of physical size 1/2x1/2.
               Indices: (6,6) to (13,13) */
      {
        for (dim = 0; dim < numDims; dim++) {
          ilower[level][dim] = 6;
          iupper[level][dim] = 13;
        }
        break;
      }
    default: {
      printf("Unknown level - cannot set grid extents\n");
      exit(1);
    }
    }
    
    //    printf("Before GridSetExtents()\n");
    /* Add this level's mesh */
    HYPRE_SStructGridSetExtents(grid, level, ilower[level], iupper[level]);
    //    printf("After GridSetExtents()\n");
    printf("  Patch Extents = ");
    printIndex(ilower[level],numDims);
    printf(" to ");
    printIndex(iupper[level],numDims);
    printf("\n");

    /* Define variable types at each part */
    HYPRE_SStructGridSetVariables(grid, level, 1, vars);
  }
  
  /* Assemble the grid */
  HYPRE_SStructGridAssemble(grid);

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the stencils\n");
  printf("----------------------------------------------------\n");
     
  /* stencil[k] = stencil at level k */
  stencils = hypre_CTAlloc(HYPRE_SStructStencil, numLevels);
  stencil_sizes = hypre_CTAlloc(int, numLevels);
  stencil_offsets = hypre_CTAlloc(Index*, numLevels);
  stencil_values = hypre_CTAlloc(double*, numLevels);

  /* Loop over stencils at different levels */
  for (level = 0; level < numLevels; level++) {
    /* 
       The following code is for general dimension.
       We use a 5-point FD discretization to L = -Laplacian in this example.
       stencil_offsets specifies the non-zero pattern of the stencil;
       Its entries are also defined here and assumed to be constant over the
       structured mesh. If not, define it later during matrix setup.
    */
    stencil_sizes[level] = 2*numDims+1;
    stencil_offsets[level] = hypre_CTAlloc(Index, stencil_sizes[level]);
    stencil_values[level] = hypre_CTAlloc(double, stencil_sizes[level]);
    entry = 0;
    /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
    /* Central coeffcient */
    for (d = 0; d < numDims; d++) {
      stencil_offsets[level][entry][d] = 0;
    }
    meshSize = h[level];
    stencil_values[level][entry] = 2*numDims/(meshSize*meshSize);
    entry++;
    for (dim = 0; dim < numDims; dim++) {
      for (side = -1; side <= 1; side += 2) {
        for (d = 0; d < numDims; d++) {
          stencil_offsets[level][entry][d] =  0;
        }
        stencil_offsets[level][entry][dim] = side;
        stencil_values[level][entry] = -1.0/(meshSize*meshSize);
        entry++;
      }
    }
    
    printf("Creating stencil ...\n");
    /* Create an empty stencil */
    HYPRE_SStructStencilCreate(numDims, stencil_sizes[level],
                               &stencils[level]);
    printf("Created stencil\n");

    /* Add stencil entries */
    printf("Level %d, Stencil offsets:\n",level);
    for (entry = 0; entry < stencil_sizes[level]; entry++) {
      printf("    entry = %d,  stencil_offsets = ",entry);
      printIndex(stencil_offsets[level][entry],numDims);
      printf("\n");
      HYPRE_SStructStencilSetEntry(stencils[level], entry,
                                   stencil_offsets[level][entry], 0);
    }
  }
  
  /*-----------------------------------------------------------
   * Set up the graph
   -----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the graph\n");
  printf("----------------------------------------------------\n");
   
  /* Create an empty graph */
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
  /* If using AMG, set graph's object type to ParCSR now */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
  }
  
  /* Set stencil (non-zero pattern) for the structured part of the grid. */
  for (level = 0; level < numLevels; level++) {
    /* Setting for variable type = 0 (cell centered). If there are more
       types, add more "GraphSetStencil" calls below. */
    HYPRE_SStructGraphSetStencil(graph, level, 0, stencils[level]);
  }
  
  /* Set unstructured part of the graph - connection between each
     consecutive levels */
  for (level = 0; level < numLevels-1; level++) {
    /*================================================================
      Boundary cells of fine level: their connection to coarse cells
      outside the fine patch
      ================================================================*/
    int flevel = level+1;
    int clevel = level;
    Index nbhrIndex;
    
    printf("### Setting level %d node connections to level %d nodes\n",
           flevel,clevel);
    /* Loop over dimensions */
    for (dim = 0; dim < numDims; dim++) {
      /* Number of cells in the fine face */
      int numCells = 1;
      for (d = 0; d < numDims; d++) {
        if (d != dim) {
          numCells *= (iupper[flevel][dim] - ilower[flevel][dim] + 1);
        }
      }
      printf("Face in the %c-dimension, numCells = %d\n",'x'+dim,numCells);
      
      for (side = -1; side <= 1; side += 2) {
        /* side= -1: left face; side = 1: right face */
        faceBoxExtents(numDims,ilower[flevel],iupper[flevel],dim,side,
                       &faceLower,&faceUpper);
        
        /* Loop over fine cells in the face */
        printf("Looping over the face: ...\n");
        int done = 0, currentDim = 0;
        Index cellIndex;
        for (d = 0; d < numDims; d++) cellIndex[d] = faceLower[d];
        while (!done) {
          printf("  cellIndex = ");
          printIndex(cellIndex,numDims);
          printf("\n");

          /*
            Check which stencil-neighbours of this cell are outside the
             fine patch.
          */
          for (entry = 0; entry < stencil_sizes[level]; entry++) {
            /* Index of fine nbhr */
            for (d = 0; d < numDims; d++) {
              nbhrIndex[d] = cellIndex[d] +
                stencil_offsets[flevel][entry][d];
            }
            printf("    entry = %d, nbhrIndex = ",entry);
            printIndex(nbhrIndex,numDims);

            for (d = 0; d < numDims; d++) {
              if ((nbhrIndex[d] < ilower[flevel][d]) ||
                  (nbhrIndex[d] > iupper[flevel][d])) {
                /* nbhr outside the fine patch */
                printf("  Needs to be eliminated");
                break;
              }
            }
            printf("\n");


            
          } // for entry
          /*
            HYPRE_SStructGraphAddEntries(graph,
            level+1, <int *index>, 0,
            level, <int *to_index>, 0);
          */
          
          cellIndex[currentDim]++;
          if (cellIndex[currentDim] > faceUpper[currentDim]) {
            while (cellIndex[currentDim] > faceUpper[currentDim]) {
              cellIndex[currentDim] = faceLower[currentDim];
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
        printf("End loop over face\n");

      } // for side

    } // for dim (face)






    /* Coarse cells outside and near fine patch: their connections to
       fine cells */
    /*
    HYPRE_SStructGraphAddEntries(graph,
                                 level, <int *index>, 0,
                                 level+1, <int *to_index>, 0);
    */
  }
  
  /* Assemble the graph */
  HYPRE_SStructGraphAssemble(graph);
  printf("nUVentries = %d\n",hypre_SStructGraphNUVEntries(graph));
  
  /*-----------------------------------------------------------
   * Set up the SStruct matrix
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Set up the SStruct matrix\n");
  printf("----------------------------------------------------\n");
   
  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  /* If using AMG, set A's object type to ParCSR now */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(A);
  
  /* Add each level's equations to A */
  printf("Adding structured equations to A\n");
  for (level = 0; level < numLevels; level++) {
    printf("At level = %d\n",level);
    /* May later need to loop over patches at this level here. */
    
    /* Set structured part entries: for each stencil entry,
       prepare a long vector with the values at all cells of an
       entry (e.g., east entry w.r.t. each cell at this level. */
    
    /* Number of cells in the mesh */
    int numCells = 1;
    for (dim = 0; dim < numDims; dim++) {
      numCells *= (iupper[level][dim] - ilower[level][dim] + 1);
    }
    values = hypre_TAlloc(double, numCells);
    printf("numCells = %d\n",numCells);

    /* Loop over stencil entries */
    for (entry = 0; entry < stencil_sizes[level]; entry++) {
      printf("add entry = %d, stencil_value = %+e, offset = ",
             entry,stencil_values[level][entry]);
      printIndex(stencil_offsets[level][entry],numDims);
      printf("\n");
      for (cell = 0; cell < numCells; cell++) {
        values[cell] = stencil_values[level][entry];
      } // for cell
      HYPRE_SStructMatrixSetBoxValues
        (A, level, ilower[level], iupper[level],
         0, 1, &entry, values);
    } // for entry

    hypre_TFree(values);

    /* Set non-stencil entries */
    /* ... Add later ... */

    /* Set boundary conditions at domain boundaries: Dirichlet (u=0) */
    printf("Setting Dirichlet B.C. at domain boundaries\n");
    for (dim = 0; dim < numDims; dim++) {
      
      /* Number of cells in the face */
      int numCells = 1;
      for (d = 0; d < numDims; d++) {
        if (d != dim) {
          numCells *= (iupper[level][dim] - ilower[level][dim] + 1);
        }
      }
      values = hypre_TAlloc(double, numCells);
      printf("Face in the %c-dimension, numCells = %d\n",'x'+dim,numCells);
      
      for (side = -1; side <= 1; side += 2) {
        /* side= -1: left face; side = 1: right face */
        printf("Face on side = %d\n",side);
        faceBoxExtents(numDims,iupper[level],iupper[level],dim,side,
                       &faceLower,&faceUpper);
        if (side < 0) {
          entry = 2*dim+1;   /* E.g. for xminus: entry=1 */
        } else {
          entry = 2*dim+2;   /* E.g. for xminus: entry=2 */
        }
        printf("Setting entry %d to 0.0 at the face nodes\n",entry);

        /* Set connections to ghost cells (outside the domain) to 0 */
        for (cell = 0; cell < numCells; cell++) {
          values[cell] = 0.0;
        } // for cell
        HYPRE_SStructMatrixSetBoxValues(A, level, faceLower, faceUpper,
                                        0, 1, &entry, values);

        /* Update central coefficient for the eliminated boundary
           condition, which is say 0.5 *(u0 + u1) = 0 in 1D. Thus
           u0 = -u1. */
        int center = 0;
        for (cell = 0; cell < numCells; cell++) {
          values[cell] = stencil_values[level][center]
            - stencil_values[level][entry];
        } // for cell
        HYPRE_SStructMatrixSetBoxValues(A, level, faceLower, faceUpper,
                                        0, 1, &center, values);

      } // for side

      hypre_TFree(values);  
    } // for dim (face)

  } // for level

  /* After talking to Barry: the following code is usually needed for an
     AMR problem: the fine patches' stencils originally extend to points that
     are outside the patch (and lie at the next-coarser level). These
     connections need to be eliminated. The connection between parts is done
     only using the C/F and F/C interfaces.
  */
  /*
     Also: A is the original composite grid operator. Residual norms in
     the solver are measured as b-A*x. 
     If FAC solver is used, it creates a new fac_A matrix from A, that is
     the original A stencils except near C/F interfaces (is it inside fine
     patches too?), but replaces them by the Galerkin R*A*P operator near
     interfaces. The grids stay the same. R,P are based on a black-box-Dendy-
     type algorithm.
     If AMG is used, make sure A is symmetric. If not, use AMG as a
     preconditioner to PCG/GMRES. Make sure the operator complexity is
     reasonable (2-3) and play with the AMG parameters. It might tend to 
     put too many gridpoints near C/F interfaces.
  */
  /* Reset matrix values so that stencil connections between two parts are
     zeroed */

  for (level = numLevels-1; level > 0; level--) {
    hypre_FacZeroCFSten(hypre_SStructMatrixPMatrix(A, level),
                        hypre_SStructMatrixPMatrix(A, level-1),
                        grid,
                        level,
                        prefinements[level]);
    hypre_FacZeroFCSten(hypre_SStructMatrixPMatrix(A, level),
                        grid,
                        level);
    hypre_ZeroAMRMatrixData(A, level-1, prefinements[level]);
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
  printf("----------------------------------------------------\n");
  printf("Set up the RHS (b), LHS (x) vectors\n");
  printf("----------------------------------------------------\n");
   
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

  //  hypre_ZeroAMRVectorData(b, plevels, prefinements);  // Do we need this?
  HYPRE_SStructVectorAssemble(b);
  //   hypre_ZeroAMRVectorData(x, plevels, prefinements);
  HYPRE_SStructVectorAssemble(x);  // See above
 
  /* For BoomerAMG solver: set up the linear system (b,x) in ParCSR format */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructVectorGetObject(b, (void **) &par_b);
    HYPRE_SStructVectorGetObject(x, (void **) &par_x);
  }
  
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
  
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Solver setup phase\n");
  printf("----------------------------------------------------\n");
   
  if (solver_id > 90) {
    /* Prepare FAC operator hierarchy using Galerkin coarsening
       with black-box interpolation, on the original meshes */
    time_fac_rap = hypre_InitializeTiming("fac rap");
    hypre_BeginTiming(time_fac_rap);
    hypre_AMR_RAP(A, prefinements, &fac_A);
    hypre_ZeroAMRVectorData(b, plevels, prefinements);
    hypre_ZeroAMRVectorData(x, plevels, prefinements);
    hypre_EndTiming(time_fac_rap);
    hypre_PrintTiming("fac rap", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_fac_rap);
    hypre_ClearTiming();
  }

  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  printf("----------------------------------------------------\n");
  printf("Print out the system and initial guess\n");
  printf("----------------------------------------------------\n");
   
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
  printf("----------------------------------------------------\n");
  printf("Solve the linear system A*x=b\n");
  printf("----------------------------------------------------\n");
   
  /*-------------- FAC Solver -----------------*/
  if (solver_id > 90) {
      n_pre  = prefinements[numLevels-1][0]-1;
      n_post = prefinements[numLevels-1][0]-1;

      /* n_pre+= n_post;*/
      /* n_post= 0;*/

      time_index = hypre_InitializeTiming("FAC Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructFACCreate(MPI_COMM_WORLD, &solver);
      HYPRE_SStructFACSetMaxLevels(solver, numLevels);
      HYPRE_SStructFACSetMaxIter(solver, 20);
      HYPRE_SStructFACSetTol(solver, 1.0e-06);
      HYPRE_SStructFACSetPLevels(solver, numLevels, plevels);
      HYPRE_SStructFACSetPRefinements(solver, numLevels, prefinements);
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
   printf("----------------------------------------------------\n");
   printf("Gather the solution vector\n");
   printf("----------------------------------------------------\n");
   
   HYPRE_SStructVectorGather(x);
   if (print_system) {
     HYPRE_SStructVectorPrint("sstruct.out.x1", x, 0);
   }
   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/
   
   if (myid == 0)
     {
       printf("\n");
       printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
   printf("----------------------------------------------------\n");
   printf("Finalize things\n");
   printf("----------------------------------------------------\n");
   
  /* Destroy grid objects */
  HYPRE_SStructGridDestroy(grid);
  hypre_TFree(ilower);
  hypre_TFree(iupper);
  hypre_TFree(h);

  /* Destroy stencil objects */
  for (level = 0; level < numLevels; level++) {
    HYPRE_SStructStencilDestroy(stencils[level]);
    hypre_TFree(stencil_offsets[level]);
    hypre_TFree(stencil_values[level]);
  }
  hypre_TFree(stencils);
  hypre_TFree(stencil_sizes);
  hypre_TFree(stencil_offsets);
  hypre_TFree(stencil_values);
  
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
   
  hypre_TFree(plevels);
  hypre_TFree(prefinements);
  //   hypre_TFree(parts);
  //   hypre_TFree(refine);
  //   hypre_TFree(distribute);
  //   hypre_TFree(block);
  
  /* Finalize MPI */
  hypre_FinalizeMemoryDebug();
  MPI_Finalize();
  
  return 0;
}
