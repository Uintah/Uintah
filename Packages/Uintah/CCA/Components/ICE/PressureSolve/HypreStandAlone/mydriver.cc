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

#include "mydriver.h"
#include "util.h"
#include "Hierarchy.h"
#include "Level.h"
#include "Patch.h"
#include "Solver.h"
#include "Param.h"
#include "TestLinear.h"

#include <vector>
#include <HYPRE_sstruct_ls.h>
#include <utilities.h>
#include <krylov.h>
#include <sstruct_mv.h>
#include <sstruct_ls.h>
#include <string>

using namespace std;

/*================== Global variables ==================*/

int     MYID;     /* The same as this proc's myid, but global */

void
makeGrid(const Param* param,
         const Hierarchy& hier,
         HYPRE_SStructGrid& grid,
         HYPRE_SStructVariable* vars)
  /*_____________________________________________________________________
    Function makeGrid:
    Create an empty Hypre grid object "grid" from our hierarchy hier,
    and add all patches from this proc to it.
    _____________________________________________________________________*/
{
  const Counter numDims   = param->numDims;
  const Counter numLevels = hier._levels.size();
  serializeProcsBegin();
  Print("Making grid\n");

  /* Add the patches that this proc owns at all levels to grid */
  for (Counter level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    Print("Level %d, meshSize = %lf, resolution = ",
          level,lev->_meshSize[0]);
    printIndex(lev->_resolution);
    fprintf(stderr,"\n");
    for (Counter i = 0; i < lev->_patchList.size(); i++) {
      Patch* patch = lev->_patchList[i];
      /* Add this patch to the grid */
      Index hypreilower, hypreiupper;
      ToIndex(patch->_ilower,&hypreilower,numDims);
      ToIndex(patch->_iupper,&hypreiupper,numDims);
      HYPRE_SStructGridSetExtents(grid, level, hypreilower, hypreiupper);
      HYPRE_SStructGridSetVariables(grid, level, NUM_VARS, vars);
      Print("  Patch %d Extents = ",i);
      printIndex(patch->_ilower);
      fprintf(stderr," to ");
      printIndex(patch->_iupper);
      fprintf(stderr,"\n");
      fflush(stderr);
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
  vector< vector<int> > stencil_offsets;

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
  stencil_offsets.resize(stencilSize);
  int entry;
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  entry = 0;
  stencil_offsets[entry].resize(numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    //    Print("Init entry = %d, dim = %d\n",entry,dim);
    stencil_offsets[entry][dim] = 0;
  }
  for (Counter dim = 0; dim < numDims; dim++) {
    for (int side = -1; side <= 1; side += 2) {
      entry++;
      stencil_offsets[entry].resize(numDims);
      //      Print("entry = %d, dim = %d\n",entry,dim);
      for (Counter d = 0; d < numDims; d++) {
        //        Print("d = %d  size = %d\n",d,stencil_offsets[entry].size());
        stencil_offsets[entry][d] = 0;
      }
      //      Print("Setting entry = %d, dim = %d\n",entry,dim);
      stencil_offsets[entry][dim] = side;
    }
  }
  
  /* Add stencil entries */
  Proc0Print("Stencil offsets:\n");
  Index hypreOffset;
  for (entry = 0; entry < stencilSize; entry++) {
    ToIndex(stencil_offsets[entry],&hypreOffset,numDims);
    HYPRE_SStructStencilSetEntry(stencil, entry, hypreOffset, 0);
    if (MYID == 0) {
      Print("    entry = %d,  stencil_offsets = ",entry);
      printIndex(stencil_offsets[entry]);
      fprintf(stderr,"\n");
      fflush(stderr);
    }
  }
  serializeProcsEnd();
}

void
makeGraph(const Param* param,
          const Hierarchy& hier,
          const HYPRE_SStructGrid& grid,
          const HYPRE_SStructStencil& stencil,
          HYPRE_SStructGraph& graph)
  /*_____________________________________________________________________
    Function makeGraph:
    Initialize the graph from stencils (interior equations) and C/F
    interface connections. Create Hypre graph object "graph" on output.
    _____________________________________________________________________*/
{
  const int numDims   = param->numDims;
  const int numLevels = hier._levels.size();
  serializeProcsBegin();
  /*
    Add structured equations (stencil-based) at the interior of
    each patch at every level to the graph.
  */
  for (Counter level = 0; level < numLevels; level++) {
    Print("  Initializing graph stencil at level %d\n",level);
    HYPRE_SStructGraphSetStencil(graph, level, 0, stencil);
  }
  
  /* 
     Add the unstructured part of the stencil connecting the
     coarse and fine level at every C/F boundary.
  */

  for (Counter level = 1; level < numLevels; level++) {
    Print("  Updating coarse-fine boundaries at level %d\n",level);
    const Level* lev = hier._levels[level];
    //    const Level* coarseLev = hier._levels[level-1];
    const vector<Counter>& refRat = lev->_refRat;

    /* Loop over patches of this proc */
    for (Counter i = 0; i < lev->_patchList.size(); i++) {
      Patch* patch = lev->_patchList[i];

      /* Loop over C/F boundaries of this patch */
      for (Counter d = 0; d < numDims; d++) {
        for (int s = -1; s <= 1; s += 2) {
          if (patch->getBoundary(d, s) == Patch::CoarseFine) {

            Print("--- Processing C/F face d = %d , s = %d ---\n",d,s);
            vector<int> faceLower(numDims);
            vector<int> faceUpper(numDims);            
            faceExtents(patch->_ilower, patch->_iupper, d, s,
                        faceLower, faceUpper);
            vector<int> coarseFaceLower(numDims);
            vector<int> coarseFaceUpper(numDims);
            int numFaceCells = 1;
            int numCoarseFaceCells = 1;
            for (Counter dd = 0; dd < numDims; dd++) {
              Print("dd = %d\n",dd);
              Print("refRat = %d\n",refRat[dd]);
              coarseFaceLower[dd] =
                faceLower[dd] / refRat[dd];
              coarseFaceUpper[dd] =
                faceUpper[dd]/ refRat[dd];
              numFaceCells       *= (faceUpper[dd] - faceLower[dd] + 1);
              numCoarseFaceCells *= (coarseFaceUpper[dd] - coarseFaceLower[dd] + 1);
            }
            Print("# fine   cell faces = %d\n",numFaceCells);
            Print("# coarse cell faces = %d\n",numCoarseFaceCells);
            //            Index* fineIndex   = new Index[numFaceCells];
            //            Index* coarseIndex = new Index[numFaceCells];
            vector<int> coarseNbhrLower = coarseFaceLower;
            vector<int> coarseNbhrUpper = coarseFaceUpper;
            coarseNbhrLower[d] += s;
            coarseNbhrUpper[d] += s;

            /*
              Loop over different types of fine cell "children" of a coarse cell
              at the C/F interface. 
            */
            vector<int> zero(numDims,0);
            vector<int> ref1(numDims,0);
            for (Counter dd = 0; dd < numDims; dd++) {
              ref1[dd] = refRat[dd] - 1;
            }
            ref1[d] = 0;
            vector<bool> activeChild(numDims,true);
            bool eocChild = false;
            vector<int> subChild = zero;
            for (Counter child = 0; !eocChild;
                 child++,
                   IndexPlusPlus(zero,ref1,activeChild,subChild,eocChild)) {
              Print("child = %4d",child);
              fprintf(stderr,"  subChild = ");
              printIndex(subChild);
              fprintf(stderr,"\n");
              vector<bool> active(numDims,true);
              bool eoc = false;
              vector<int> subCoarse = coarseNbhrLower;
              for (Counter cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,
                                   active,subCoarse,eoc)) {
                Print("  cell = %4d",cell);
                fprintf(stderr,"  subCoarse = ");
                printIndex(subCoarse);
                vector<int> subFine(numDims);
                /* Compute fine cell inside the fine patch from coarse
                   cell outside the fine patch */
                subFine = subCoarse;
                if (s < 0) {
                  /* Left boundary: go from coarse outside to coarse
                     inside the patch, then find its lower left corner
                     and that's fine child 0. */
                  subFine[d] -= s;
                  pointwiseMult(refRat,subFine,subFine);
                } else {
                  /* Right boundary: start from the coarse outside the
                     fine patch, find its lower left corner, then find
                     its fine nbhr inside the fine patch. This is fine
                     child 0. */
                  pointwiseMult(refRat,subFine,subFine);
                  subFine[d] -= s;
                }
                pointwiseAdd(subFine,subChild,subFine);
                fprintf(stderr,"  subFine = ");
                printIndex(subFine);
                fprintf(stderr,"\n");
                Index hypreSubFine, hypreSubCoarse;
                ToIndex(subFine,&hypreSubFine,numDims);
                ToIndex(subCoarse,&hypreSubCoarse,numDims);
              
                /* Add the connections between the fine and coarse nodes
                   to the graph */
                Print("  Adding connection to graph\n");
                HYPRE_SStructGraphAddEntries(graph,
                                             level,   hypreSubFine,   0,
                                             level-1, hypreSubCoarse, 0); 
                HYPRE_SStructGraphAddEntries(graph,
                                             level-1, hypreSubCoarse, 0,
                                             level,   hypreSubFine,   0);
              } // end for cell

            } // end for child
            //            delete[] fineIndex;
            //            delete[] coarseIndex;
          } // end if boundary is CF interface
        } // end for s
      } // end for d
    } // end for i (patches)
  } // end for level
  serializeProcsEnd();

  /* Assemble the graph */
  HYPRE_SStructGraphAssemble(graph);
  Print("Assembled graph, nUVentries = %d\n",
        hypre_SStructGraphNUVEntries(graph));
}

int
main(int argc, char *argv[]) {
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/
  /* Initialize parameters */
  Param* param = new TestLinear;

  /* Set test cast parameters */
  param->numDims       = 2; //3;
  param->solverID      = 30;    // solver ID. 30 = AMG, 99 = FAC
  param->numLevels     = 2;     // Number of AMR levels
  param->baseResolution= 8;     // Level 0 grid size in every direction
  param->printSystem   = true;

  int                   numProcs, myid;
  int                   time_index;

  /* Grid data structures */
  HYPRE_SStructGrid     grid;
  Hierarchy             hier(param);
  
  /* Stencil data structures. We use the same stencil all all levels
     and all parts. */
  HYPRE_SStructStencil  stencil;

  /* Graph data structures */
  HYPRE_SStructGraph    graph;
  
  /* Solver data structures */
  Solver                solver(param);
  
  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  param->numProcs = numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MYID = myid;
#if DEBUG
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
    if ((param->solverID > 90) && ((numLevels < 2) || (numDims != 3))) {
      fprintf(stderr,"FAC solver needs a 3D problem and at least 2 levels.");
      clean();
      exit(1);
    }
    fprintf(stderr,"done\n");

    Print("Checking # procs ... ");
    //    int correct = mypow(2,numDims);
    int correct = int(pow(2.0,numDims));
    if (numProcs != correct) {
      Print("\n\nError, hard coded to %d processors in %d-D for now.\n",
            correct,numDims);
      clean();
      exit(1);
    }
    fprintf(stderr,"numProcs = %d, done\n",numProcs);

    Print("\n");
  }

  time_index = hypre_InitializeTiming("SStruct Interface");
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

  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars
  /* Create an empty grid in numDims dimensions with # parts = numLevels. */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);
  makeGrid(param, hier, grid, vars);             // Make Hypre grid from hier
  hier.printPatchBoundaries();

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the stencils on all the patchs\n");
  Proc0Print("----------------------------------------------------\n");
  makeStencil(param, hier, stencil);

  /*-----------------------------------------------------------
   * Set up the graph
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the graph\n");
  Proc0Print("----------------------------------------------------\n");
  /* Create an empty graph */
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
  /* If using AMG, set graph's object type to ParCSR now */
  if ( ((param->solverID >= 20) && (param->solverID <= 30)) ||
       ((param->solverID >= 40) && (param->solverID < 60)) ) {
    HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
  }

  makeGraph(param, hier, grid, stencil, graph);

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
  solver.initialize(hier, grid, stencil, graph);

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
  solver.printMatrix("output_A");
  solver.printRHS("output_b");
  solver.printSolution("output_x0");

  /*-----------------------------------------------------------
   * Solve the linear system A*x=b
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Solve the linear system A*x=b\n");
  Proc0Print("----------------------------------------------------\n");
  solver.solve();

  /*-----------------------------------------------------------
   * Print the solution and other info
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Print the solution vector\n");
  Proc0Print("----------------------------------------------------\n");
  solver.printSolution("output_x1");
  Proc0Print("Iterations = %d\n", solver._results.numIterations);
  Proc0Print("Final Relative Residual Norm = %e\n",
             solver._results.finalResNorm);
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
   
  /* Destroy graph objects */
  Print("Destroying graph objects\n");
  HYPRE_SStructGraphDestroy(graph);
   
  clean();

  Print("%s: Going down successfully\n",argv[0]);

  return 0;
} // end main()
