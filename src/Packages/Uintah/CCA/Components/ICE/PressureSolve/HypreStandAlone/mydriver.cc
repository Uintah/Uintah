/*--------------------------------------------------------------------------
 * File: mydriver.cc
 *
 * Test driver for semi-structured matrix interface.
 * This is a stand-alone hypre interface that uses FAC / AMG solvers to solve
 * the pressure equation in implicit AMR-ICE.
 *
 * Revision history:
 * 19-JUL-2005   Dav & Oren   Works (empty though) for 4 procs, does not crash.
 *--------------------------------------------------------------------------*/
#include "mydriver.h"
#include <vector>
using namespace std;

#include "krylov.h"
#include "sstruct_mv.h"
#include "sstruct_ls.h"
 
#define DEBUG    1
#define MAX_DIMS 3
#define NUM_VARS 1
int     numDims;
int     MYID;  /* The same as myid, but global */
typedef int Index[MAX_DIMS];

class Part {
 public:
  int         _procID;
  int         _levelID;
  int         _partID;
  vector<int> _ilower;
  vector<int> _iupper;
  
  Part(const int procID, 
       const int levelID,
       const int partID) 
  { _procID = procID; _levelID = levelID; _partID = partID; }
  Part(const int procID, 
       const int levelID,
       const int partID,
       const vector<int>& ilower, const vector<int>& iupper)
  { _procID = procID; _levelID = levelID; _partID = partID;
  _ilower = ilower; _iupper = iupper; }

 private:
};

class Level {
public:
  vector<double> _meshSize;
  vector<Part*>  _partList;
 
  Level(const double h) {
    _meshSize.resize(numDims);
    for (int d = 0; d < numDims; d++) {
      _meshSize[d] = h;
    }
  }

private:
};

class Hierarchy {
public:
  vector<Level*> _levels;
  unsigned int   _numParts;
  Hierarchy(void) { _numParts = 0; }
};

#if 0
void loopHypercube(vector<int> ilower, vector<int> iupper,
                   vector<int> step, vector<int>* list) {
  /*_____________________________________________________________________
    Function loopHypercube:
    Prepare a list of all cell indices in the hypercube specified by
    ilower,iupper, as if we're looping over them (numDims nested loops)
    with step size step. The output list of indices is return in list.
    list has to be allocated outside this function.
    _____________________________________________________________________*/
  int done = 0, currentDim = 0, count = 0, d;
  vector<int> cellIndex;
  for (d = 0; d < numDims; d++) cellIndex[d] = ilower[d];
  while (!done) {
    Print("  count = %2d  cellIndex = ",count);
    printIndex(cellIndex);
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
#endif

void
makeGrid(const int numProcs,
         const Hierarchy& hier,
         HYPRE_SStructGrid& grid,
         HYPRE_SStructVariable* vars)
  /*_____________________________________________________________________
    Function makeGrid:
    Synchronize processors and update the part IDs so that they are
    consecutive (across processors as well as within processor and level).
    Create an empty grid object "grid" and put all parts from all procs
    in it.
    _____________________________________________________________________*/
{
  Print("numProcs = %d\n",numProcs);
  int* numPartsIn = new int[numProcs];
  int* numPartsOut = new int[numProcs];

  for (int index = 0; index < numProcs; index++) {
    numPartsIn[index] = 0;
    numPartsOut[index] = 0;
  }
  numPartsIn[MYID] = hier._numParts;
  MPI_Allreduce(numPartsIn, numPartsOut, numProcs,
                MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (MYID == 0) {
    for (int index = 0; index < numProcs; index++) {
      Print("numPartsIn[%d] = %d     numPartsOut[%d] = %d\n",
            index,numPartsIn[index],index,numPartsOut[index]);
    }
    Print("\n");
  }
 
  int sumParts = 0;  // Number of parts of all procs with id < this proc's id
  for (int index = 0; index < MYID; index++) {
    sumParts += numPartsOut[index];
  }
  int totParts = 0;  // Total number of parts
  for (int index = 0; index < numProcs; index++) {
    totParts += numPartsOut[index];
  }

  /* Create an empty grid in numDims dimensions with numParts parts
     (=patches). */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, totParts, &grid);

  /* 
     Update partIDs so that they are globally consecutive, 
     and add them to grid.
  */
  int numLevels = hier._levels.size();
  Print("Number of levels = %d\n",numLevels);
  for (int level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    Print("Level %d, meshSize = %lf\n",level,lev->_meshSize[0]);
    for (int i = 0; i < lev->_partList.size(); i++) {
      Part* part = lev->_partList[i];
      int& partID = part->_partID;
      partID += sumParts;
      /* Add this patch to the grid */
      Index hypreilower, hypreiupper;
      ToIndex(part->_ilower,&hypreilower);
      ToIndex(part->_iupper,&hypreiupper);
      HYPRE_SStructGridSetExtents(grid, partID, hypreilower, hypreiupper);
      HYPRE_SStructGridSetVariables(grid, partID, NUM_VARS, vars);
      Print("  Part %d Extents = ",partID);
      printIndex(part->_ilower);
      printf(" to ");
      printIndex(part->_iupper);
      printf("\n");
      fflush(stdout);
    }
  }

  /*
    Assemble the grid; this is a collective call that synchronizes
    data from all processors. On exit from this function, the grid is
    ready.
  */
  HYPRE_SStructGridAssemble(grid);
  if (MYID == 0) {
    Print("\n");
    Print("Assembled grid num parts %d\n", hypre_SStructGridNParts(grid));
    Print("\n");
  }

  delete numPartsIn;
  delete numPartsOut;
}

int main(int argc, char *argv[]) {
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/
  /* Counters, specific sizes of this problem */
  int numProcs, myid;
  int solver_id = 30; // solver ID. 30 = AMG, 99 = FAC
  int numLevels = 2;  // Number of AMR levels
  int n         = 4;  // Level 0 grid size in every direction
   numDims = 3;

  /* Grid data structures */
  HYPRE_SStructGrid   grid;
  Hierarchy           hier;
  Index               faceLower, faceUpper;
  int                 *levelID;  // Needed by Hypre FAC 
  Index               *refinementRatio; // Needed by Hypre FAC 
  
  /* Stencil data structures.
     We use the same stencil all all levels and all parts. */
  HYPRE_SStructStencil stencil;
  int                 stencil_size;
  vector< vector<int> > stencil_offsets;
  
  /* Graph data structures */
  HYPRE_SStructGraph  graph, fac_graph;
  
  /* For matrix setup */
  //  Index index, to_index;
  double *values; //, *box_values;

  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  HYPRE_SStructMatrix A, fac_A;
  HYPRE_SStructVector b;
  HYPRE_SStructVector x;
  HYPRE_SStructSolver solver;
  HYPRE_ParCSRMatrix  par_A;
  HYPRE_ParVector     par_b;
  HYPRE_ParVector     par_x;
  HYPRE_Solver        par_solver;
  int                 num_iterations, n_pre, n_post;
  double              final_res_norm;
  
  /* Timers, debugging flags */
  int                 time_index, time_fac_rap;
  int                 print_system = 1;
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MYID = myid;
#if DEBUG
  hypre_InitMemoryDebug(myid);
#endif

  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  for (int i = 0; i < myid; i++) {
    //    Print("Beginning Barrier # %d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
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
    //    int correct = mypow(2,numDims);
    int correct = int(pow(2.0,numDims));
    if (numProcs != correct) {
      Print("\n\nError, hard coded to %d processors in %d-D for now.\n",
            correct,numDims);
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

  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars

  /* Initialize arrays holding level and parts info */
  levelID         = hypre_TAlloc(int, numLevels);    
  refinementRatio = hypre_TAlloc(Index, numLevels);

#if 0
  int procMap[4][2] = { 
    {0,0}, {0,1}, {1,0}, {1,1} 
  }; // Works for 2D; write general gray code
#endif
#if 1
  int procMap[8][3] = { 
    {0,0,0}, {0,1,0}, {1,0,0}, {1,1,0} ,
    {0,0,1}, {0,1,1}, {1,0,1}, {1,1,1} 
  }; // Works for 3D; write general gray code
#endif

  /* Initialize the parts at all levels that this proc owns */
  for (int level = 0; level < numLevels; level++) {
    levelID[level] = level;   /* Level ID */
    
    /* Refinement ratio w.r.t. parent level. Assumed to be constant
       (1:2) in all dimensions and all levels for now. */
    if (level == 0) {
      /* Dummy value */
      for (int dim = 0; dim < numDims; dim++) {
        refinementRatio[level][dim] = 0;
      }
    } else {
      for (int dim = 0; dim < numDims; dim++) {
        refinementRatio[level][dim] = 2;
      }
    }
    
    double h;
    /* Compute meshsize, assumed the same in all directions at each level */
    if (level == 0) {
      h = 1.0/n;   // Size of domain divided by # of gridpoints
    } else {
      h = hier._levels[level-1]->_meshSize[0] / 
        refinementRatio[level][0]; // ref. ratio constant for all dims
    }
    
    hier._levels.push_back(new Level(h));
    Level* lev = hier._levels[level];

    /* Mesh box extents (lower left corner, upper right corner) */
    switch (level)
      {
      case 0:
        /* Level 0 extends over the entire domain. */
        {
          vector<int> ilower(numDims);
          vector<int> iupper(numDims);
          for (int dim = 0; dim < numDims; dim++) {
            ilower[dim] = procMap[myid][dim] * n;
            iupper[dim] = ilower[dim] + n - 1;
          }
          int partID = hier._numParts;
          hier._numParts++;
          Part* part = new Part(myid,level,partID,ilower,iupper);
          lev->_partList.push_back(part);
          break;
        }
      case 1:
        /* Level 1 extends over central square of physical size 1/2x1/2. */
        {
          vector<int> ilower(numDims);
          vector<int> iupper(numDims);
          for (int dim = 0; dim < numDims; dim++) {
            ilower[dim] = n + procMap[myid][dim] * n;
            iupper[dim] = ilower[dim] + n - 1;
          }
          int partID = hier._numParts;
          hier._numParts++;
          Part* part = new Part(myid,level,partID,ilower,iupper);
          lev->_partList.push_back(part);
          break;
        }
      default:
        {
          Print("Unknown level - cannot set grid extents\n");
          clean();
          exit(1);
        }
      }
  }
  
  for (int i = numProcs-1; i >= myid; i--) {
    Print("End of Grid Barrier # %d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
  makeGrid(numProcs, hier, grid, vars);

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  for (int i = 0; i < myid; i++) {
    //    Print("Beginning Barrier # %d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
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
  //  Print("stencil_size = %d   numDims = %d\n",stencil_size,numDims);
  stencil_offsets.resize(stencil_size);
  int entry;
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  entry = 0;
  stencil_offsets[entry].resize(numDims);
  for (int dim = 0; dim < numDims; dim++) {
    //    Print("Init entry = %d, dim = %d\n",entry,dim);
    stencil_offsets[entry][dim] = 0;
  }
  for (int dim = 0; dim < numDims; dim++) {
    for (int side = -1; side <= 1; side += 2) {
      entry++;
      stencil_offsets[entry].resize(numDims);
      //      Print("entry = %d, dim = %d\n",entry,dim);
      for (int d = 0; d < numDims; d++) {
        //        Print("d = %d  size = %d\n",d,stencil_offsets[entry].size());
        stencil_offsets[entry][d] = 0;
      }
      //      Print("Setting entry = %d, dim = %d\n",entry,dim);
      stencil_offsets[entry][dim] = side;
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
  Index hypreOffset;
  for (entry = 0; entry < stencil_size; entry++) {
    ToIndex(stencil_offsets[entry],&hypreOffset);
    HYPRE_SStructStencilSetEntry(stencil, entry, hypreOffset, 0);
    if (myid == 0) {
      Print("    entry = %d,  stencil_offsets = ",entry);
      printIndex(stencil_offsets[entry]);
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
  for (int level = 0; level < numLevels; level++) {
    Print("  Initializing graph stencil at level %d\n",level);
    Level* lev = hier._levels[level];
    for (int i = 0; i < lev->_partList.size(); i++) {
      int partID = lev->_partList[i]->_partID;
      HYPRE_SStructGraphSetStencil(graph, partID, 0, stencil);
    }
  }

  /* Assemble the graph */
  for (int i = numProcs-1; i >= myid; i--) {
    //    Print("End of Grid Barrier # %d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
  HYPRE_SStructGraphAssemble(graph);
  Print("Assembled graph, nUVentries = %d\n",
        hypre_SStructGraphNUVEntries(graph));
  
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
  Print("Created empty SStructMatrix\n");
  /* If using AMG, set A's object type to ParCSR now */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(A);

  /*=== TESTING BEGIN ===*/
  if (myid == 0) {
    for (int level = 0; level < numLevels; level++) {
      Level* lev = hier._levels[level];
      for (int i = 0; i < lev->_partList.size(); i++) {
        Part* part = lev->_partList[i];
        int& partID = part->_partID;
        //      Index hypreilower, hypreiupper;
        //      ToIndex(part->_ilower,&hypreilower);
        //      ToIndex(part->_iupper,&hypreiupper);
        Print("  Part %d Extents = ",partID);
        printIndex(part->_ilower);
        printf(" to ");
        printIndex(part->_iupper);
        printf("\n");
        fflush(stdout);
        Print("Looping over cells in this patch:\n");
        vector<int> sub = part->_ilower;
        vector<bool> active(numDims);
        for (int d = 0; d < numDims; d++) active[d] = true;
        active[1] = false;
        bool eof = false;
        for (int cell = 0; !eof;
             cell++,
               IndexPlusPlus(part->_ilower,part->_iupper,active,sub,eof)) {
          Print("cell = %4d",cell);
          printf("  sub = ");
          printIndex(sub);
          printf("  eof = %d\n",eof);
        }
      }
    }
  }
    /*=== TESTING END ===*/




  // Add here interior equations of each part to A
#if 0
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
#endif
  /* Assemble the matrix */
  HYPRE_SStructMatrixAssemble(A);
  
  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if ( ((solver_id >= 20) && (solver_id <= 30)) ||
       ((solver_id >= 40) && (solver_id < 60)) ) {
    HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
  }
#if 0
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
#endif
  /* Print total time for setting up the linear system */
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
  Print("End timing\n");

#if 0
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
#endif
  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
   if (myid == 0) {
     Print("----------------------------------------------------\n");
     Print("Finalize things\n");
     Print("----------------------------------------------------\n");
   }

   /* Destroy grid objects */
   Print("Destroying grid objects\n");
   HYPRE_SStructGridDestroy(grid);
   hypre_TFree(levelID);
   hypre_TFree(refinementRatio);
   
   /* Destroy stencil objects */
   Print("Destroying stencil objects\n");
   HYPRE_SStructStencilDestroy(stencil);
   
   /* Destroy graph objects */
   Print("Destroying graph objects\n");
   if (solver_id > 90) {
     fac_graph = hypre_SStructMatrixGraph(fac_A);
     HYPRE_SStructGraphDestroy(fac_graph);
   }
   HYPRE_SStructGraphDestroy(graph);
   
   /* Destroy matrix, RHS, solution objects */
   Print("Destroying matrix, RHS, solution objects\n");
   if (solver_id > 90) {
     HYPRE_SStructMatrixDestroy(fac_A);
   }
   HYPRE_SStructMatrixDestroy(A);
   //   HYPRE_SStructVectorDestroy(b);
   //   HYPRE_SStructVectorDestroy(x);
   
   Print("Cleaning\n");
   clean();

   Print("%s: Going down successfully\n",argv[0]);
   return 0;
} // end main()
