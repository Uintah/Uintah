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
#include "Param.h"
#include "Hierarchy.h"
#include "Level.h"
#include "Patch.h"
#include "util.h"
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
getPatchesFromOtherProcs( const Param & param, const Hierarchy & hier )
{
  int sendbuf[param.numProcs];  /* Suppose this proc is #0 has 5 patches,
                                   don't know what everyone else has.
                                   So this array will look like : 5 0 0 0 0. */
  int numPatches[param.numProcs];  // After assembling sendbuf from all procs

  for( int level = 0; level < hier._levels.size(); level++ ) {
    Level* lev = hier._levels[level];
    const vector<int>& resolution = lev->_resolution;
    // clear sendbuf
    for( int index = 0; index < param.numProcs; index++ ) {
      if( index == MYID ) {
        sendbuf[index] = lev->_patchList.size();
      } else {
        sendbuf[index] = 0;
      }
    }

    // Talk to all procs to find out how many patches they have on
    // this level.
    MPI_Allreduce(sendbuf, numPatches, param.numProcs,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int totalPatches = 0;
    int startPosition = 0;
    for(int index = 0; index < param.numProcs; index++ ) {
      //      Proc0Print("has %d patches on level %d\n",
      // numPatches[index], level);
      totalPatches += numPatches[index];
      if( index < MYID )
        startPosition += numPatches[index];
    }

    //    Proc0Print("got totalPatches of %d\n", totalPatches);

    // Put our patch information into a big vector, share it with
    // all other procs
    int recordSize = 2*param.numDims+1;
    int * sendPatchInfo = new int[ recordSize * totalPatches ];
    int * patchInfo     = new int[ recordSize * totalPatches ];
    for( int index = 0; index < recordSize*totalPatches; index++ ) {
      sendPatchInfo[index] = 0;
      patchInfo[index] = -1;
    }
    int count = startPosition * recordSize;
    for(int index = 0; index < lev->_patchList.size(); index++ ) {
      Patch* patch = lev->_patchList[index];
      sendPatchInfo[count++] = patch->_procID;
      for (int d = 0; d < param.numDims; d++)
        sendPatchInfo[count++] = patch->_ilower[d];
      for (int d = 0; d < param.numDims; d++)
        sendPatchInfo[count++] = patch->_iupper[d];
    }

    MPI_Allreduce(sendPatchInfo, patchInfo, totalPatches*recordSize,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    //    for(int index = 0; index < recordSize*totalPatches; index++ ) {
    // Proc0Print("%3d: %3d %3d\n", index,sendPatchInfo[index],patchInfo[index]);
    //}
    //    Print("Looping over patches and setting their boundary types\n");
    vector<int> ilower(param.numDims);
    vector<int> iupper(param.numDims);
    vector<int> otherilower(param.numDims);
    vector<int> otheriupper(param.numDims);
    int patchIndex = 0;

    for (int index = 0; index < totalPatches; index++ ) {
      int owner = patchInfo[recordSize*index];
      if (MYID != owner) {   // This patch is processed only on its owning proc
        continue;
      }
      Patch* patch = lev->_patchList[patchIndex];
      patchIndex++;
      for (int d = 0; d < param.numDims; d++) {
        ilower[d] = patchInfo[recordSize*index + d + 1];
        iupper[d] = patchInfo[recordSize*index + param.numDims + d + 1];

        /* Default: boundary is a C/F boundary */
        for (int s = -1; s <= 1; s += 2) {
          patch->setBoundary(d,s,Patch::CoarseFine);
        }
        
        /* Check if patch is near a domain boundary */
        if (ilower[d] == 0) {
          patch->setBoundary(d,-1,Patch::Domain);
        }

        if (iupper[d] == resolution[d]-1) {
          patch->setBoundary(d,1,Patch::Domain);
        }
      }

      for (int other = 0; other < totalPatches; other++) {
        if (other == index)
          continue;
        for (int d = 0; d < param.numDims; d++) {
          otherilower[d] = patchInfo[recordSize*other + d + 1];
          otheriupper[d] = patchInfo[recordSize*other + param.numDims + d + 1];
        }
        /*
        Print("Comparing patch index=%d: from ",index);
        printIndex(ilower);
        fprintf(stderr," to ");
        printIndex(iupper);
        fprintf(stderr,"  owned by proc %d",patchInfo[recordSize*index]);
        fprintf(stderr,"\n");
        Print("To patch other=%d: from ",other);
        printIndex(otherilower);
        fprintf(stderr," to ");
        printIndex(otheriupper);
        fprintf(stderr,"  owned by proc %d",patchInfo[recordSize*other]);
        fprintf(stderr,"\n");
        */
        for (int d = 0; d < param.numDims; d++) {
          /* Check if patch has a nbhring patch on its left */
          if (ilower[d] == otheriupper[d]+1) {
            for (int d2 = 0; d2 < param.numDims; d2++) {
              if (d2 == d) continue;
              if (max(ilower[d2],otherilower[d2]) <=
                  min(iupper[d2],otheriupper[d2])) {
                patch->setBoundary(d,-1,Patch::Neighbor);
              }
            }
          }

          /* Check if patch has a nbhring patch on its right */
          if (iupper[d] == otherilower[d]-1) {
            for (int d2 = 0; d2 < param.numDims; d2++) {
              if (d2 == d) continue;
              if (max(ilower[d2],otherilower[d2]) <=
                  min(iupper[d2],otheriupper[d2])) {
                patch->setBoundary(d,1,Patch::Neighbor);
              }
            }
          }
        }
      } // end for other
    } // end for index
    delete [] sendPatchInfo;
    delete [] patchInfo;
  } // end for level
}

void
printPatchBoundaries( const Param & param, const Hierarchy & hier )
{
  serializeProcsBegin();
  /* Print boundary types */
  for( int level = 0; level < hier._levels.size(); level++ ) {
    Print("---- Patch boundaries at level %d ----\n",level);
    Level* lev = hier._levels[level];
    for (int index = 0; index < lev->_patchList.size(); index++ ) {
      Patch* patch = lev->_patchList[index];
      Print("Patch # %d: from ",index);
      printIndex(patch->_ilower);
      fprintf(stderr," to ");
      printIndex(patch->_iupper);
      fprintf(stderr,"\n");
      for (int d = 0; d < param.numDims; d++) {
        for (int s = -1; s <= 1; s += 2) {
          //          Print("  boundary( d = %d , s = %+d ) = %s\n",
          //                d,s,boundaryTypeString[patch->getBoundary(d,s)].c_str());
          Print("  boundary( d = %d , s = %+d ) = %s\n",
                d,s,Patch::boundaryTypeString[patch->getBoundary(d,s)].c_str());
        }
      }
    }
  } // end for level
  Print("\n");
  serializeProcsEnd();
} // end getPatchesFromOtherProcs()

void
makeHierarchy(const Param& param,
              Hierarchy& hier,
              int *plevel,
              Index *refinementRatio)
  /*_____________________________________________________________________
    Function makeHierarchy:
    Create a static refinement hierarchy hier into our data structures of
    Hierarchy, Levels and Patches. We define here all the patches owned
    by this proc only. Initialize the plevel and refinementRatio arrays
    needed by Hypre FAC. p = strcture of input parameters.
    _____________________________________________________________________*/
{
  const int numProcs  = param.numProcs;
  const int numDims   = param.numDims;
  const int numLevels = param.numLevels;
  const int n         = param.baseResolution;

  vector<int> k(numDims,2);
  IntMatrix procMap = grayCode(numDims,k);
  /* Print procMap */
  for(int i = 0;i < procMap.numRows();i++){
    for(int j = 0; j < procMap.numCols(); j++){
      if (j == 0) {
        Proc0Print("\t%d",procMap(i,j));
      } else {
        if (MYID == 0) {
          fprintf(stderr,"\t%d",procMap(i,j));
        }
      }
    }
    if (MYID == 0) {
      fprintf(stderr,"\n");
    }
  }

  /* Initialize the patches that THIS proc owns at all levels */
  for (int level = 0; level < numLevels; level++) {
    plevel[level] = level;   // part ID of this level
    vector<int> refRat(numDims);
    /* Refinement ratio w.r.t. parent level. Assumed to be constant
       (1:2) in all dimensions and all levels for now. */
    if (level == 0) {        // Dummy ref. ratio value */
      for (int dim = 0; dim < numDims; dim++) {
        refRat[dim] = 1;
      }
    } else {
      for (int dim = 0; dim < numDims; dim++) {
        refRat[dim] = 2;
      }
    }
    ToIndex(refRat,&refinementRatio[level],numDims);

    double h;
    /* Compute meshsize, assumed the same in all directions at each level */
    if (level == 0) {
      h = 1.0/n;   // Size of domain divided by # of gridpoints
    } else {
      h = hier._levels[level-1]->_meshSize[0] / 
        refRat[0]; // ref. ratio constant for all dims
    }
    
    hier._levels.push_back(new Level(numDims,h));
    Level* lev = hier._levels[level];
    lev->_refRat = refRat;
    vector<int> ilower(numDims);
    vector<int> iupper(numDims);

    /* Mesh box extents (lower left corner, upper right corner) */
    for (int dim = 0; dim < numDims; dim++) {
      if( level == 0 ) {
        ilower[dim] = procMap(MYID,dim) * n/2;
        iupper[dim] = ilower[dim] + n/2 - 1;
      } else if( level == 1 ) {
        ilower[dim] = n/2 + procMap(MYID,dim) * n/2;
        iupper[dim] = ilower[dim] + n/2 - 1;
      } else {
        fprintf(stderr,"Unknown level\n");
        clean();
        exit(1);
      }
    }
    Patch* patch = new Patch(MYID,level,ilower,iupper);
    lev->_patchList.push_back(patch);

  } // end for level
}

void
makeGrid(const Param& param,
         const Hierarchy& hier,
         HYPRE_SStructGrid& grid,
         HYPRE_SStructVariable* vars)
  /*_____________________________________________________________________
    Function makeGrid:
    Create an empty Hypre grid object "grid" from our hierarchy hier,
    and add all patches from this proc to it.
    _____________________________________________________________________*/
{
  const int numDims   = param.numDims;
  const int numLevels = hier._levels.size();
  serializeProcsBegin();
  Print("Making grid\n");

  /* Add the patches that this proc owns at all levels to grid */
  for (int level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    Print("Level %d, meshSize = %lf, resolution = ",
          level,lev->_meshSize[0]);
    printIndex(lev->_resolution);
    fprintf(stderr,"\n");
    for (int i = 0; i < lev->_patchList.size(); i++) {
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
makeStencil(const Param& param,
            const Hierarchy& hier,
            HYPRE_SStructStencil& stencil)
  /*_____________________________________________________________________
    Function makeStencil:
    Initialize the Hypre stencil with a 5-point stencil (in d dimensions).
    Create Hypre stencil object "stencil" on output.
    _____________________________________________________________________*/
{
  const int numDims   = param.numDims;
  const int numLevels = hier._levels.size();
  int                   stencilSize = 2*numDims+1;
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
makeGraph(const Param& param,
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
  const int numDims   = param.numDims;
  const int numLevels = hier._levels.size();
  serializeProcsBegin();
  /*
    Add structured equations (stencil-based) at the interior of
    each patch at every level to the graph.
   */
  for (int level = 0; level < numLevels; level++) {
    Print("  Initializing graph stencil at level %d\n",level);
    HYPRE_SStructGraphSetStencil(graph, level, 0, stencil);
  }
  
  /* 
     Add the unstructured part of the stencil connecting the
     coarse and fine level at every C/F boundary.
  */

  for (int level = 1; level < numLevels; level++) {
    Print("  Updating coarse-fine boundaries at level %d\n",level);
    const Level* lev = hier._levels[level];
    //    const Level* coarseLev = hier._levels[level-1];
    const vector<int>& refRat = lev->_refRat;

    /* Loop over patches of this proc */
    for (int i = 0; i < lev->_patchList.size(); i++) {
      Patch* patch = lev->_patchList[i];

      /* Loop over C/F boundaries of this patch */
      for (int d = 0; d < numDims; d++) {
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
            for (int dd = 0; dd < numDims; dd++) {
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
            for (int dd = 0; dd < numDims; dd++) {
              ref1[dd] = refRat[dd] - 1;
            }
            ref1[d] = 0;
            vector<bool> activeChild(numDims,true);
            bool eocChild = false;
            vector<int> subChild = zero;
            for (int child = 0; !eocChild;
                 child++,
                   IndexPlusPlus(zero,ref1,activeChild,subChild,eocChild)) {
              Print("child = %4d",child);
              fprintf(stderr,"  subChild = ");
              printIndex(subChild);
              fprintf(stderr,"\n");
              vector<bool> active(numDims,true);
              bool eoc = false;
              vector<int> subCoarse = coarseNbhrLower;
              for (int cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,active,subCoarse,eoc)) {
                Print("  cell = %4d",cell);
                fprintf(stderr,"  subCoarse = ");
                printIndex(subCoarse);
                vector<int> subFine(numDims);
                subFine = subCoarse;
                subFine[d] -= s;
                pointwiseMult(subFine,refRat,subFine);
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

void
makeLinearSystem(const Param& param,
                 const Hierarchy& hier,
                 const HYPRE_SStructGrid& grid,
                 const HYPRE_SStructStencil& stencil,
                 const HYPRE_SStructGraph& graph,
                 HYPRE_SStructMatrix& A,
                 HYPRE_SStructVector& b)
  /*_____________________________________________________________________
    Function makeLinearSystem:
    Initialize the linear system: set up the values on the links of the
    graph of the LHS matrix A and value of the RHS vector b at all
    patches of all levels. Delete coarse data underlying fine patches.
    _____________________________________________________________________*/
{
  serializeProcsBegin();
  const int numDims   = param.numDims;
  const int numLevels = hier._levels.size();
  /*
    Add equations at all interior cells of every patch owned by this proc
    to A. Eliminate boundary conditions at domain boundaries.
  */
  Print("Adding interior equations to A\n");
  vector<double> xCell(numDims);
  vector<double> xNbhr(numDims);
  vector<double> xFace(numDims);
  vector<double> offset(numDims);
  vector<double> coarseOffset(numDims);
  vector<int> sub;
  vector<bool> active(numDims);
  int stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];
  for (int entry = 0; entry < stencilSize; entry++) entries[entry] = entry;
  for (int level = 0; level < numLevels; level++) {
    const Level* lev = hier._levels[level];
    const vector<double>& h = lev->_meshSize;
    const vector<int>& resolution = lev->_resolution;
    scalarMult(h,0.5,offset);
    double cellVolume = prod(h);
    for (int i = 0; i < lev->_patchList.size(); i++) {
      /* Add equations of interior cells of this patch to A */
      Patch* patch = lev->_patchList[i];
      double* values    = new double[stencilSize * patch->_numCells];
      double* rhsValues = new double[patch->_numCells];
      Index hypreilower, hypreiupper;
      ToIndex(patch->_ilower,&hypreilower,numDims);
      ToIndex(patch->_iupper,&hypreiupper,numDims);
      Print("  Adding interior equations at Patch %d, Extents = ",i);
      printIndex(patch->_ilower);
      fprintf(stderr," to ");
      printIndex(patch->_iupper);
      fprintf(stderr,"\n");
      fflush(stderr);
      Print("Looping over cells in this patch:\n");
      sub = patch->_ilower;        
      for (int d = 0; d < numDims; d++) active[d] = true;
      bool eoc = false;
      for (int cell = 0; !eoc;
           cell++,
             IndexPlusPlus(patch->_ilower,patch->_iupper,active,sub,eoc)) {
        Print("cell = %4d",cell);
        fprintf(stderr,"  sub = ");
        printIndex(sub);
        fprintf(stderr,"\n");
        int offsetValues    = stencilSize * cell;
        int offsetRhsValues = cell;
        /* Initialize the stencil values of this cell's equation to 0 */
        for (int entry = 0; entry < stencilSize; entry++) {
          values[offsetValues + entry] = 0.0;
        }
        rhsValues[offsetRhsValues] = 0.0;
        
        /* Loop over directions */
        int entry = 1;
        for (int d = 0; d < numDims; d++) {
          double faceArea = cellVolume / h[d];
          for (int s = -1; s <= 1; s += 2) {
            Print("--- d = %d , s = %d, entry = %d ---\n",d,s,entry);
            /* Compute coordinates of:
               This cell's center: xCell
               The neighboring's cell data point: xNbhr
               The face crossing between xCell and xNbhr: xFace
            */
            pointwiseMult(sub,h,xCell);
            pointwiseAdd(xCell,offset,xCell);
            
            xNbhr    = xCell;
            xNbhr[d] += s*h[d];
            if ((sub[d] == 0) && (s == -1) ||
                (sub[d] == resolution[d]-1) && (s ==  1)) {
              xNbhr[d] = xCell[d] + 0.5*s*h[d];
            }
            xFace    = xCell;
            xFace[d] = 0.5*(xCell[d] + xNbhr[d]);

            Print("xCell = ");
            printIndex(xCell);
            fprintf(stderr," xNbhr = ");
            printIndex(xNbhr);
            fprintf(stderr," xFace = ");
            printIndex(xFace);
            fprintf(stderr,"\n");

            /* Compute the harmonic average of the diffusion
               coefficient */
            double a    = 1.0; // Assumed constant a for now
            double diff = fabs(xNbhr[d] - xCell[d]);
            double flux = a * faceArea / diff;

            /* Accumulate this flux'es contributions in values */
            values[offsetValues        ] += flux;
            values[offsetValues + entry] -= flux;
            if ((sub[d] == 0) && (s == -1) ||
                (sub[d] == resolution[d]-1) && (s ==  1)) {
              /* Nbhr is at the boundary, eliminate it from values */
              values[offsetValues + entry] = 0.0;
            }

            rhsValues[offsetRhsValues] = 0.0;  // Assuming 0 source term
            entry++;
          } // end for s
        } // end for d

        /*======== BEGIN GOOD DEBUGGING CHECK =========*/
        /* This will set the diagonal entry of this cell's equation
           to cell so that we can compare our cell numbering with
           Hypre's cell numbering within each patch.
           Hypre does it like we do: first loop over x, then over y,
           then over z. */
        /*        values[offsetValues] = cell; */
        /*======== END GOOD DEBUGGING CHECK =========*/

      } // end for cell
      
      /* Print values, rhsValues vectors */
      for (int cell = 0; cell < patch->_numCells; cell++) {
        int offsetValues    = stencilSize * cell;
        int offsetRhsValues = cell;
        Print("cell = %4d\n",cell);
        for (int entry = 0; entry < stencilSize; entry++) {
          Print("values   [%5d] = %+.3f\n",
                offsetValues + entry,values[offsetValues + entry]);
        }
        Print("rhsValues[%5d] = %+.3f\n",
              offsetRhsValues,rhsValues[offsetRhsValues]);
        Print("-------------------------------\n");
      } // end for cell

      /* Add this patch to the Hypre structure for A */
      HYPRE_SStructMatrixSetBoxValues(A, level, 
                                      hypreilower, hypreiupper, 0,
                                      stencilSize, entries, values);
      delete[] values;
      delete[] rhsValues;
    } // end for patch
  } // end for level

  delete entries;
  Print("Done adding interior equations\n");

  Print("Begin deleting underlying coarse data\n");
  /* 
     Set the values on the graph links of the unstructured part
     connecting the coarse and fine level at every C/F boundary.
  */

  for (int level = 1; level < numLevels; level++) {
    Print("  Updating coarse-fine boundaries at level %d\n",level);
    const Level* lev = hier._levels[level];
    const vector<int>& refRat = lev->_refRat;
    const vector<double>& h = lev->_meshSize;
    scalarMult(h,0.5,offset);
    double cellVolume = prod(h);

    const Level* coarseLev = hier._levels[level-1];
    const vector<double>& coarseH = coarseLev->_meshSize;
    scalarMult(coarseH,0.5,coarseOffset);
    double coarseCellVolume = prod(coarseH);

    /* Loop over patches of this proc */
    for (int i = 0; i < lev->_patchList.size(); i++) {
      Patch* patch = lev->_patchList[i];

      /* Loop over C/F boundaries of this patch */
      //      for (int d = 0; d < numDims; d++) {
      for (int d = 1; d <= 1; d++) { // TESTING
        double faceArea = cellVolume / h[d];
        double coarseFaceArea = coarseCellVolume / coarseH[d];

        //        for (int s = -1; s <= 1; s += 2) {
        for (int s = -1; s <= -1; s++) { // TESTING
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
            for (int dd = 0; dd < numDims; dd++) {
              Print("dd = %d\n",dd);
              Print("refRat = %d\n",refRat[dd]);
              coarseFaceLower[dd] =
                faceLower[dd] / refRat[dd];
              coarseFaceUpper[dd] =
                faceUpper[dd]/ refRat[dd];
              numFaceCells       *= (faceUpper[dd] - faceLower[dd] + 1);
              numCoarseFaceCells *= (coarseFaceUpper[dd] - coarseFaceLower[dd]
                                     + 1);
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
              Loop over different types of fine cell "children" of a
              coarse cell at the C/F interface. 
            */
            vector<int> zero(numDims,0);
            vector<int> ref1(numDims,0);
            for (int dd = 0; dd < numDims; dd++) {
              ref1[dd] = refRat[dd] - 1;
            }
            ref1[d] = 0;
            vector<bool> activeChild(numDims,true);
            bool eocChild = false;
            vector<int> subChild = zero;
            for (int child = 0; !eocChild;
                 child++,
                   IndexPlusPlus(zero,ref1,activeChild,subChild,eocChild)) {
              Print("child = %4d",child);
              fprintf(stderr,"  subChild = ");
              printIndex(subChild);
              fprintf(stderr,"\n");
              vector<bool> active(numDims,true);
              bool eoc = false;
              vector<int> subCoarse = coarseNbhrLower;
              for (int cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,active,
                                   subCoarse,eoc)) {
                Print("  cell = %4d",cell);
                fprintf(stderr,"  subCoarse = ");
                printIndex(subCoarse);
                vector<int> subFine(numDims);
                subFine = subCoarse;
                subFine[d] -= s;
                pointwiseMult(subFine,refRat,subFine);
                pointwiseAdd(subFine,subChild,subFine);
                fprintf(stderr,"  subFine = ");
                printIndex(subFine);
                fprintf(stderr,"\n");
                Index hypreSubFine, hypreSubCoarse;
                ToIndex(subFine,&hypreSubFine,numDims);
                ToIndex(subCoarse,&hypreSubCoarse,numDims);
              
                /* Set the weights on the connections between the fine
                   and coarse nodes in A */
                Print("    Adding C/F flux to matrix\n");

                /*====================================================
                  Add the flux from the fine child cell to the
                  coarse nbhr cell.
                  ====================================================*/
                /* Compute coordinates of:
                   This cell's center: xCell
                   The neighboring's cell data point: xNbhr
                   The face crossing between xCell and xNbhr: xFace
                */
                int side = -s;
                pointwiseMult(subFine,h,xCell);
                pointwiseAdd(xCell,offset,xCell);   
                
                pointwiseMult(subCoarse,coarseH,xNbhr);
                pointwiseAdd(xNbhr,coarseOffset,xNbhr);    
                /* xFace is a convex combination of xCell, xNbhr with
                   weights depending on the distances of the coarse
                   and fine meshsizes, i.e., their distances in the d
                   dimension from the C/F boundary. */
                double alpha = coarseH[d] / (coarseH[d] + h[d]);
                for (int dd = 0; dd < numDims; dd++) {
                  xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
                }
                Print("    xCell = ");
                printIndex(xCell);
                fprintf(stderr," xNbhr = ");
                printIndex(xNbhr);
                fprintf(stderr," xFace = ");
                printIndex(xFace);
                fprintf(stderr,"\n");

                /* Compute the harmonic average of the diffusion
                   coefficient */
                double a    = 1.0; // Assumed constant a for now
                double diff = 0;
                for (int dd = 0; dd < numDims; dd++) {
                  diff += pow(xNbhr[dd] - xCell[dd],2);
                }
                diff = sqrt(diff);
                double flux = a * faceArea / diff;
                Print("    C/F flux = %f   a = %f  face = %f  diff = %f\n",
                      flux,a,faceArea,diff);
                
                /* Add the C-F flux to the fine cell equation - stencil part */
                const int numStencilEntries = 1;
                int stencilEntries[numStencilEntries] = {0};
                double stencilValues[numStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(A,
                                               level, hypreSubFine,     0,
                                               numStencilEntries,
                                               stencilEntries,
                                               stencilValues);

                /* Add the C-F flux to the fine cell equation - graph part */
                const int numGraphEntries = 1;
                int graphEntries[numGraphEntries] = {stencilSize};
                double graphValues[numGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(A,
                                               level, hypreSubFine,     0,
                                               numGraphEntries,
                                               graphEntries,
                                               graphValues);

                /* Subtract the C-F flux from the coarse cell equation -
                   stencil part */
                const int numCoarseStencilEntries = 1;
                int coarseStencilEntries[numCoarseStencilEntries] = {0};
                double coarseStencilValues[numCoarseStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(A,
                                               level-1, hypreSubCoarse,  0,
                                               numCoarseStencilEntries,
                                               coarseStencilEntries,
                                               coarseStencilValues);

                /* Subtract the C-F flux from the coarse cell equation -
                   graph part */
                const int numCoarseGraphEntries = 1;
                int coarseGraphEntries[numCoarseGraphEntries] = {stencilSize};
                double coarseGraphValues[numCoarseGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(A,
                                               level-1, hypreSubCoarse,     0,
                                               numCoarseGraphEntries,
                                               coarseGraphEntries,
                                               coarseGraphValues);

                /*
                  Set the reverse connection; coincidentally, it uses
                  the same graphEntries value as in the previous
                  call.
                 */
                HYPRE_SStructMatrixSetValues(A,
                                             level-1, hypreSubCoarse,   0,
                                             1, graphEntries, graphValues);


                if (child == 0) { /* Potential source for bugs: remove
                                     coarse-coarse flux only ONCE in the
                                     child loop. */
                 Print("  Removing C/C flux connections from matrix\n");
                 /*====================================================
                    Remove the flux from the coarse nbhr to its coarse
                    stencil nbhr that underlies the fine patch.
                    ====================================================*/
                  /* Compute coordinates of:
                     This cell's center: xCell
                     The neighboring's cell data point: xNbhr
                     The face crossing between xCell and xNbhr: xFace
                  */
                  int side = -s;
                  pointwiseMult(subCoarse,coarseH,xCell);
                  pointwiseAdd(xCell,coarseOffset,xCell);               
                  xNbhr    = xCell;
                  xNbhr[d] += side*coarseH[d];
                  xFace    = xCell;
                  xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
                  Print("xCell = ");
                  printIndex(xCell);
                  fprintf(stderr," xNbhr = ");
                  printIndex(xNbhr);
                  fprintf(stderr," xFace = ");
                  printIndex(xFace);
                  fprintf(stderr,"\n");

                  /* Compute the harmonic average of the diffusion
                     coefficient */
                  double a    = 1.0; // Assumed constant a for now
                  double diff = fabs(xNbhr[d] - xCell[d]);
                  double flux = a * coarseFaceArea / diff;
                  Print("C/C flux = %f   a = %f  face = %f  diff = %f\n",
                        flux,a,coarseFaceArea,diff);

                  const int coarseNumStencilEntries = 2;
                  int coarseStencilEntries[coarseNumStencilEntries] =
                    {0, 2*d + ((-s+1)/2) + 1};
                  double coarseStencilValues[coarseNumStencilEntries] =
                    {-flux, flux};
                  HYPRE_SStructMatrixAddToValues(A,
                                                 level-1, hypreSubCoarse, 0,
                                                 coarseNumStencilEntries,
                                                 coarseStencilEntries,
                                                 coarseStencilValues);
                } // if (child == 0)
              } // end for cell

            } // end for child
            //            delete[] fineIndex;
            //            delete[] coarseIndex;
          } // end if boundary is CF interface
        } // end for s
      } // end for d
    } // end for i (patches)
  } // end for level





#if 0
  /* We will implement the following ourselves, because it does not work
     in problems that are not in 3-D.
  */
  /* 
     Zero out all the connections from fine point stencils to outside the
     fine patch. These are replaced by the graph connections between the fine
     patch and its parent coarse patch.
  */
  for (int level = numLevels-1; level > 0; level--) {
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
  Print("End deleting underlying coarse data\n");
  serializeProcsEnd();
}

int
main(int argc, char *argv[]) {
  /*-----------------------------------------------------------
   * Variable definition, parameter init, arguments verification
   *-----------------------------------------------------------*/
  /* Counters, parameters and specific sizes of this problem */
  int numProcs, myid;
  Param param;
  param.numDims       = 2;
  param.solverID      = 30;    // solver ID. 30 = AMG, 99 = FAC
  param.numLevels     = 2;     // Number of AMR levels
  param.baseResolution= 8;     // Level 0 grid size in every direction
  param.printSystem   = true;

  /* Grid data structures */
  HYPRE_SStructGrid     grid;
  Hierarchy             hier;
  Index                 faceLower, faceUpper;
  int                   *plevel;          // Needed by FAC: part # of level
  Index                 *refinementRatio; // Needed by FAC
  
  /* Stencil data structures.
     We use the same stencil all all levels and all parts. */
  HYPRE_SStructStencil  stencil;

  /* Graph data structures */
  HYPRE_SStructGraph    graph, fac_graph;
  
  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  HYPRE_SStructMatrix   A, fac_A;
  HYPRE_SStructVector   b, x;
  HYPRE_SStructSolver   solver;
  HYPRE_ParCSRMatrix    par_A;
  HYPRE_ParVector       par_b;
  HYPRE_ParVector       par_x;
  HYPRE_Solver          par_solver;
  int                   num_iterations, n_pre, n_post;
  double                final_res_norm;
  
  /* Timers, debugging flags */
  int                   time_index, time_fac_rap;
  
  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  param.numProcs = numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MYID = myid;
#if DEBUG
  hypre_InitMemoryDebug(myid);
#endif
  const int numLevels = param.numLevels;
  const int numDims   = param.numDims;

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
    if ((param.solverID > 90) && ((numLevels < 2) || (numDims != 3))) {
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
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the grid (AMR levels, patches)\n");
    Print("----------------------------------------------------\n");
  }

  /* Initialize arrays needed by Hypre FAC */
  plevel          = hypre_TAlloc(int  , numLevels);    
  refinementRatio = hypre_TAlloc(Index, numLevels);

  makeHierarchy(param, hier, plevel, refinementRatio);// Set our MR hierarchy
  getPatchesFromOtherProcs(param, hier);  // Compute global patch IDs
  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars
  /* Create an empty grid in numDims dimensions with # parts = numLevels. */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);
  makeGrid(param, hier, grid, vars);             // Make Hypre grid from hier
  printPatchBoundaries(param, hier);

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the stencils on all the patchs\n");
    Print("----------------------------------------------------\n");
  }
  makeStencil(param, hier, stencil);

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
  if ( ((param.solverID >= 20) && (param.solverID <= 30)) ||
       ((param.solverID >= 40) && (param.solverID < 60)) ) {
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
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Set up the SStruct matrix\n");
    Print("----------------------------------------------------\n");
  }
  
  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  Print("Created empty SStructMatrix\n");
  /* If using AMG, set A's object type to ParCSR now */
  if ( ((param.solverID >= 20) && (param.solverID <= 30)) ||
       ((param.solverID >= 40) && (param.solverID < 60)) ) {
    HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(A);

  /* Initialize RHS vector b and solution vector x */
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
  /* If AMG is used, set b and x type to ParCSR */
  if ( ((param.solverID >= 20) && (param.solverID <= 30)) ||
       ((param.solverID >= 40) && (param.solverID < 60)) ) {
    HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(b);
  HYPRE_SStructVectorInitialize(x);

  makeLinearSystem(param, hier, grid, stencil, graph, A, b);
  
  /* Assemble the matrix - a collective call */
  HYPRE_SStructMatrixAssemble(A); 
  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if ( ((param.solverID >= 20) && (param.solverID <= 30)) ||
       ((param.solverID >= 40) && (param.solverID < 60)) ) {
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
  

  /* Initialize b at all levels */
  fprintf(stderr,"Adding structured equations to b and x\n");
  for (level = 0; level < numLevels; level++) {
    fprintf(stderr,"At level = %d\n",level);
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
    fprintf(stderr,"numCells = %d\n",numCells);

    fprintf(stderr,"-- Initializing b\n");
    for (cell = 0; cell < numCells; cell++) {
      values[cell] = 1.0;   /* RHS value at cell */
    } // for cell
    HYPRE_SStructVectorSetBoxValues(b, level, ilower[level], iupper[level],
                                    0, values);

    fprintf(stderr,"-- Initializing x\n");
    for (cell = 0; cell < numCells; cell++) {
      values[cell] = 1.0;   /* Initial guess for LHS - value at cell */
    } // for cell
    HYPRE_SStructVectorSetBoxValues(x, level, ilower[level], iupper[level],
                                    0, values);
    
    hypre_TFree(values);
    
  } // for level

  hypre_ZeroAMRVectorData(b, plevel, refinementRatio);  // Do we need this?
  HYPRE_SStructVectorAssemble(b);
  hypre_ZeroAMRVectorData(x, plevel, refinementRatio);
  HYPRE_SStructVectorAssemble(x);  // See above
 
  /* For BoomerAMG solver: set up the linear system (b,x) in ParCSR format */
  if ( ((param.solverID >= 20) && (param.solverID <= 30)) ||
       ((param.solverID >= 40) && (param.solverID < 60)) ) {
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
  if (param.solverID > 90) {
    /* FAC Solver. Prepare FAC operator hierarchy using Galerkin coarsening
       with black-box interpolation, on the original meshes */
    time_fac_rap = hypre_InitializeTiming("fac rap");
    hypre_BeginTiming(time_fac_rap);
    hypre_AMR_RAP(A, refinementRatio, &fac_A);
    hypre_ZeroAMRVectorData(b, plevel, refinementRatio);
    hypre_ZeroAMRVectorData(x, plevel, refinementRatio);
    hypre_EndTiming(time_fac_rap);
    hypre_PrintTiming("fac rap", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_fac_rap);
    hypre_ClearTiming();
  }
#endif

  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Print out the system and initial guess\n");
    Print("----------------------------------------------------\n");
  }
  if (param.printSystem) {
    HYPRE_SStructMatrixPrint("sstruct.out.A", A, 0);
    if (param.solverID > 90) {
      HYPRE_SStructMatrixPrint("sstruct.out.facA", fac_A, 0);
    }
    if (param.solverID == 30) {
      HYPRE_ParCSRMatrixPrint(par_A, "sstruct.out.parA");
      /* Print CSR matrix in IJ format, base 1 for rows and cols */
      HYPRE_ParCSRMatrixPrintIJ(par_A, 1, 1, "sstruct.out.ijA");
    }
    //    HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
    //    HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
  }

#if 0
  /*-----------------------------------------------------------
   * Solve the linear system A*x=b
   *-----------------------------------------------------------*/
  if (myid == 0) {
    Print("----------------------------------------------------\n");
    Print("Solve the linear system A*x=b\n");
    Print("----------------------------------------------------\n");
  }

  /*-------------- FAC Solver -----------------*/
  if (param.solverID > 90) {
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
    HYPRE_SStructFACSetPLevels(solver, numLevels, plevel);
    HYPRE_SStructFACSetPRefinements(solver, numLevels, refinementRatio);
    HYPRE_SStructFACSetRelChange(solver, 0);
    if (param.solverID > 90)
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

    if (param.solverID > 90)
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
  if (param.solverID == 30)
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
  hypre_TFree(plevel);
  hypre_TFree(refinementRatio);
   
  /* Destroy stencil objects */
  Print("Destroying stencil objects\n");
  HYPRE_SStructStencilDestroy(stencil);
   
  /* Destroy graph objects */
  Print("Destroying graph objects\n");
  if (param.solverID > 90) {
    fac_graph = hypre_SStructMatrixGraph(fac_A);
    HYPRE_SStructGraphDestroy(fac_graph);
  }
  HYPRE_SStructGraphDestroy(graph);
   
  /* Destroy matrix, RHS, solution objects */
  Print("Destroying matrix, RHS, solution objects\n");
  if (param.solverID > 90) {
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
