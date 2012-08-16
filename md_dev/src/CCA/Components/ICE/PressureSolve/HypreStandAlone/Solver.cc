/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include "Solver.h"

#include "util.h"
#include "Level.h"
#include "Patch.h"

#include <string>
#include <map>

using namespace std;

void
Solver::initialize(const Hierarchy& hier,
                   const HYPRE_SStructGrid& grid,
                   const HYPRE_SStructStencil& stencil)
{
  funcPrint("Solver::initialize()",FBegin);
  _requiresPar = 
    (((_solverID >= 20) && (_solverID <= 30)) ||
     ((_solverID >= 40) && (_solverID < 60)));
  dbg0 << "requiresPar = " << _requiresPar << "\n";

  makeGraph(hier, grid, stencil);
  initializeData(hier, grid);
  makeLinearSystem(hier, grid, stencil);
  assemble();
  funcPrint("Solver::initialize()",FEnd);
}

void
Solver::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid)
{
  funcPrint("Solver::initializeData()",FBegin);

  // Create an empty matrix with the graph non-zero pattern
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, _graph, &_A);
  dbg0 << "Created empty SStructMatrix" << "\n";
  // Initialize RHS vector b and solution vector x
  dbg << "Create empty b,x" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  dbg << "Done b" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  dbg << "Done x" << "\n";

  // If using AMG, set (A,b,x)'s object type to ParCSR now
  if (_requiresPar) {
    dbg0 << "Matrix object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
    dbg << "Vector object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    dbg << "Done b" << "\n";
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
    dbg << "Done x" << "\n";
  }
  dbg << "Init A" << "\n";
  HYPRE_SStructMatrixInitialize(_A);
  dbg << "Init b,x" << "\n";
  HYPRE_SStructVectorInitialize(_b);
  dbg << "Done b" << "\n";
  HYPRE_SStructVectorInitialize(_x);
  dbg << "Done x" << "\n";

  funcPrint("Solver::initializeData()",FEnd);
}

void
Solver::assemble(void)
{
  dbg << "Solver::assemble() end" << "\n";
  // Assemble the matrix - a collective call
  HYPRE_SStructMatrixAssemble(_A); 
  HYPRE_SStructVectorAssemble(_b);
  HYPRE_SStructVectorAssemble(_x);

  // For BoomerAMG solver: set up the linear system in ParCSR format
  if (_requiresPar) {
    HYPRE_SStructMatrixGetObject(_A, (void **) &_parA);
    HYPRE_SStructVectorGetObject(_b, (void **) &_parB);
    HYPRE_SStructVectorGetObject(_x, (void **) &_parX);
  }
  dbg << "Solver::assemble() begin" << "\n";
}

void
Solver::setup(void)
{
  //-----------------------------------------------------------
  // Solver setup phase
  //-----------------------------------------------------------
  linePrint("#",60);
  dbg0 << "Solver setup phase" << "\n";
  linePrint("#",60);
  int time_index = hypre_InitializeTiming("AMG Setup");
  hypre_BeginTiming(time_index);
  
  this->setup(); // which setup will this be? The derived class's?
  
  hypre_EndTiming(time_index);
  hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
}

void
Solver::solve(void)
{
  //-----------------------------------------------------------
  // Solver solve phase
  //-----------------------------------------------------------
  linePrint("#",60);
  dbg0 << "Solver solve phase" << "\n";
  linePrint("#",60);

  this->solve(); // which setup will this be? The derived class's?

  //-----------------------------------------------------------
  // Gather the solution vector
  //-----------------------------------------------------------
  // TODO: SolverSStruct is derived from Solver; implement the following
  // in SolverSStruct. For SolverStruct (PFMG), another gather vector required.
  linePrint("#",60);
  dbg0 << "Gather the solution vector" << "\n";
  linePrint("#",60);

  HYPRE_SStructVectorGather(_x);
} //end solve()

/*================= GRAPH CONSTRUCTION FUNCTIONS =========================*/
void
Solver::makeConnections(const ConstructionStatus& status,
                        const Hierarchy& hier,
                        const HYPRE_SStructStencil& stencil,
                        const Counter level,
                        const Patch* patch,
                        const Counter& d,
                        const Side& s,
                        const CoarseFineViewpoint& viewpoint)
  // Build the C/F connections at the (d,s) C/F face of patch "patch"
  // at level "level" (connecting to level-1). We add fine-to-coarse
  // connections if viewpoint = FineToCoarse, otherwise we add
  // coarse-to-fine connections.
{
  linePrint("=",50);
  dbg0 << "Building connections" 
       << "  level = " << level 
       << "  patch =\n" << *patch << "\n"
       << "  status = " << status
       << "  viewpoint = " << viewpoint << "\n"
       << "  Face d = " << d
       << " , s = " << s << "\n";
  linePrint("=",50);
  dbg.setLevel(2);
  const Counter numDims = _param->numDims;

  // Level info initialization
  Counter fineLevel, coarseLevel;
  if (viewpoint == FineToCoarse) {
    fineLevel = level;
    coarseLevel = level-1;
  } else { // viewpoint == CoarseToFine
    coarseLevel = level;
    fineLevel = level+1;
  }

  const Level* lev = hier._levels[fineLevel];
  const Vector<Counter>& refRat = lev->_refRat;
  const Vector<double>& h = lev->_meshSize;
  Vector<double> fineOffset = 0.5 * h;
  double cellVolume = h.prod();
  double faceArea = cellVolume / h[d];

  const Level* coarseLev = hier._levels[coarseLevel];
  const Vector<double>& coarseH = coarseLev->_meshSize;
  Vector<double> coarseOffset = 0.5 * coarseH;
  double coarseCellVolume = coarseH.prod();
  double coarseFaceArea = coarseCellVolume / coarseH[d];

  // Stencil info initialization
  Counter stencilSize = hypre_SStructStencilSize(stencil);

  // Fine cells of this face
  Box faceFineBox = patch->_box.faceExtents(d,s);
  // Coarse cells on the other side of the C/F boundary
  Box faceCoarseBox = faceFineBox.coarseNbhrExtents(refRat,d,s);

  dbg.setLevel(2);
  dbg << "coarseLevel = " << coarseLevel << "\n";
  dbg << "fineLevel   = " << fineLevel   << "\n";
  // Loop over the C/F coarse cells and add connections
  for(Box::iterator coarse_iter = faceCoarseBox.begin();
      coarse_iter != faceCoarseBox.end(); ++coarse_iter) {
    // Compute the part fineCellFace of the fine face that directly
    // borders this coarse cell.
    Vector<int> cellFaceLower;
    if (s == Left) {
      Vector<int> coarseCellOverFineCells = *coarse_iter;
      coarseCellOverFineCells[d] -= s;
      cellFaceLower = coarseCellOverFineCells * refRat;
    } else { // s == Right
      Vector<int> coarseCellOverFineCells = *coarse_iter;
      cellFaceLower = coarseCellOverFineCells * refRat;
      cellFaceLower[d] -= s;
    }
    Vector<int> offset = refRat - 1;
    offset[d] = 0;
    Vector<int> cellFaceUpper = cellFaceLower + offset;
    Box fineCellFace(cellFaceLower,cellFaceUpper);

    // Loop over the fine cells in fineCellFace and add their
    // connections to graph/matrix
    dbg.indent();
    bool removeCCconnection = true; // C-C connection is removed once
                                    // per the loop over the fine cells below
    Counter fineCell = 0;
    for (Box::iterator fine_iter = fineCellFace.begin();
         fine_iter != fineCellFace.end(); ++fine_iter, ++fineCell) {
      dbg.setLevel(2);
      dbg << "Coarse cell: " << *coarse_iter << "\n";
      dbg << "Fine   cell: " << *fine_iter   << "\n";
      if (status == Graph) {
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Add F->C connections to graph
          //====================================================
          dbg << "Adding F->C connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       fineLevel,(*fine_iter).getData(),0,
                                       coarseLevel,(*coarse_iter).getData(),0);
          dbg << "HYPRE call done" << "\n";
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Add C->F connections to graph
          //====================================================
          dbg << "Adding C->F connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       coarseLevel,(*coarse_iter).getData(),0,
                                       fineLevel,(*fine_iter).getData(),0);
          dbg << "HYPRE call done" << "\n";
        } // end if viewpoint
      } else { // status == Matrix
        
        //########################################################
        // Compute C-F interface flux.
        // Compute coordinates of:
        // This cell's center: xCell
        // The neighboring's cell data point: xNbhr
        // The face crossing between xCell and xNbhr: xFace
        //########################################################
        // xCell at this level, xNbhr at coarser level
        Vector<double> xCell = fineOffset + (h * (*fine_iter));
        Vector<double> xNbhr = coarseOffset + (coarseH * (*coarse_iter));
        // xFace is a convex combination of xCell, xNbhr with
        // weights depending on the distances of the coarse and fine
        // meshsizes, i.e., their distances in the d dimension from
        // the C/F boundary.
        double alpha = coarseH[d] / (coarseH[d] + h[d]);
        dbg.setLevel(3);
        dbg << "alpha = " << alpha << "\n";
        Vector<double> xFace(0,numDims);
        for (Counter dd = 0; dd < numDims; dd++) {
          xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
        }
        dbg << "xCell = " << xCell
            << ", xFace = " << xFace
            << ", xNbhr = " << xNbhr << "\n";
        
        //########################################################
        // Compute the harmonic average of the diffusion coefficient
        //########################################################
        double a    = 1.0; // Assumed constant a for now
        double diff = 0;
        for (Counter dd = 0; dd < numDims; dd++) {
          diff += pow(xNbhr[dd] - xCell[dd],2);
        }
        diff = sqrt(diff);
        double flux = a * faceArea / diff;
        dbg << "C/F flux = " << flux
            << "   a = " << a
            << " face = " << faceArea
            << " diff = " << diff << "\n";
        
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Compute matrix entries of the F->C graph connections
          //====================================================
          dbg << "Adding F->C flux to matrix" << "\n";
                    
          //########################################################
          // Add the flux to the fine cell equation - stencil part
          //########################################################
          const int numStencilEntries = 1;
          int stencilEntries[numStencilEntries] = {0};
          double stencilValues[numStencilEntries] = {flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         fineLevel, (*fine_iter).getData(),
                                         0, numStencilEntries,
                                         stencilEntries,
                                         stencilValues);

          //########################################################
          // Add the C-F flux to the fine cell equation - graph part
          //########################################################
          const int numGraphEntries = 1;
          Counter entry = stencilSize;
          // Find the correct graph entry corresponding to this coarse
          // nbhr (subCoarse) of the fine cell subFine
          for (Counter dd = 0; dd < d; dd++) {
            // Are we near the ss-side boundary and is it a C/F bdry?
            for (Side ss = Left; ss <= Right; ++ss) {
              if ((patch->getBoundaryType(dd,ss) == Patch::CoarseFine) &&
                  ((*fine_iter)[dd] == patch->_box.get(ss)[dd])) {
                entry++;
              }
            }
          }
          if ((s == Right) &&
              (patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
              ((*fine_iter)[d] == patch->_box.get(Left)[d])) {
            entry++;
          }
          dbg.setLevel(2);
          dbg << "entry (Fine cell -> coarse) = " << entry << "\n";
          int graphEntries[numGraphEntries] = {entry};
          double graphValues[numGraphEntries] = {-flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         fineLevel,(*fine_iter).getData(),
                                         0,numGraphEntries,
                                         graphEntries,
                                         graphValues);
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Compute matrix entries of the C->F graph connections
          //====================================================
          dbg << "Adding C->F flux to matrix" << "\n";

          //########################################################
          // Add the C/F flux coarse cell equation - stencil part
          //########################################################
          const int numCoarseStencilEntries = 1;
          int coarseStencilEntries[numCoarseStencilEntries] = {0};
          double coarseStencilValues[numCoarseStencilEntries] = {flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         coarseLevel,(*coarse_iter).getData()
                                         ,0,numCoarseStencilEntries,
                                         coarseStencilEntries,
                                         coarseStencilValues);

          //########################################################
          // Add the C/F flux coarse cell equation - graph part
          //########################################################
          const int numCoarseGraphEntries = 1;
          int coarseGraphEntries[numCoarseGraphEntries] =
            {stencilSize+fineCell};
          dbg << "fineCell = " << fineCell << "\n";
          dbg << "entry (coarse cell -> fine cell) = "
              << stencilSize+fineCell << "\n";
          double coarseGraphValues[numCoarseGraphEntries] = {-flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         coarseLevel, (*coarse_iter).getData(), 0,
                                         numCoarseGraphEntries,
                                         coarseGraphEntries,
                                         coarseGraphValues);
          if (removeCCconnection) {
            //########################################################
            // Remove the flux from the coarse nbhr to its coarse
            // stencil nbhr that underlies the fine patch.
            //########################################################
            // Compute coordinates of:
            // This cell's center: xCell
            // The neighboring's cell data point: xNbhr
            // The face crossing between xCell and xNbhr: xFace
            removeCCconnection = false;
            dbg << "Removing C/C flux connections from matrix" << "\n";
            Side s2 = Side(-s);
            dbg << "      s = " << s
                << ", s2 = " << s2 << "\n";
            xCell = coarseOffset + (coarseH * (*coarse_iter));
            xNbhr    = xCell;
            xNbhr[d] += s2*coarseH[d];
            xFace    = xCell;
            xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
            dbg << "xCell = " << xCell
                << ", xFace = " << xFace
                << ", xNbhr = " << xNbhr << "\n";
            
            // Compute the harmonic average of the diffusion
            // coefficient
            double a    = 1.0; // Assumed constant a for now
            double diff = fabs(xNbhr[d] - xCell[d]);
            double flux = a * coarseFaceArea / diff;
            dbg << "      C/F flux = " << flux
                << "   a = " << a
                << " face = " << faceArea
                << " diff = " << diff << "\n";
            const int coarseNumStencilEntries = 2;
            int coarseStencilEntries[coarseNumStencilEntries] =
              {0, 2*d + ((-s+1)/2) + 1};
            double coarseStencilValues[coarseNumStencilEntries] =
              {-flux, flux};
            HYPRE_SStructMatrixAddToValues(_A,
                                           coarseLevel,(*coarse_iter).getData(), 0,
                                           coarseNumStencilEntries,
                                           coarseStencilEntries,
                                           coarseStencilValues);
          } // end if (fineCell == 0)
        } // end if viewpoint
      } // end if status
    } // end for fine_iter
    dbg.unindent();
  } // end for coarse_iter
} // end makeConnections

void
Solver::makeGraph(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeGraph~
  // Initialize the graph from stencils (interior equations) and C/F
  // interface connections. Create Hypre graph object "_graph" on output.
  //_____________________________________________________________________
{
  serializeProcsBegin();
  funcPrint("Solver::makeGraph()",FBegin);
  const Counter numDims = _param->numDims;
  //-----------------------------------------------------------
  // Set up the graph
  //-----------------------------------------------------------
  linePrint("#",60);
  dbg0 << "Set up the graph" << "\n";
  linePrint("#",60);
  serializeProcsEnd();

  // Create an empty graph
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &_graph);
  // If using AMG, set graph's object type to ParCSR now
  if (_requiresPar) {
    dbg0 << "graph object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  //  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  //======================================================================
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  dbg0 << "Graph structured (interior) connections" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg0 << "  Initializing graph stencil at level " << level
         << " of " << numLevels << "\n";
    HYPRE_SStructGraphSetStencil(_graph, level, 0, stencil);
  }
  serializeProcsEnd();

  //======================================================================
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  dbg0 << "Graph unstructured (C/F) connections" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg.setLevel(1);
    const Level* lev = hier._levels[level];
    dbg.indent();
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      dbg.setLevel(2);
      linePrint("$",40);
      dbg << "Processing Patch" << "\n"
          << *patch << "\n";
      linePrint("$",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        linePrint("=",50);
        dbg0 << "Building fine-to-coarse connections" << "\n";
        linePrint("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              dbg.setLevel(2);
              dbg << "boundary is " << patch->getBoundaryType(d,s) << "\n";
              makeConnections(Graph,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        linePrint("=",50);
        dbg0 << "Building coarse-to-fine connections" << "\n";
        linePrint("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        dbg.indent();
        // List of fine patches covering this patch
        vector<Patch*> finePatchList = hier.finePatchesOverMe(*patch);
        Box coarseRefined(patch->_box.get(Left) * refRat,
                          (patch->_box.get(Right) + 1) * refRat - 1);
        dbg << "coarseRefined " << coarseRefined << "\n";
    
        //===================================================================
        // Loop over next-finer level patches that cover this patch
        //===================================================================
        for (vector<Patch*>::iterator iter = finePatchList.begin();
             iter != finePatchList.end(); ++iter) {
          //===================================================================
          // Compute relevant boxes at coarse and fine levels
          //===================================================================
          Patch* finePatch = *iter;
          dbg.setLevel(3);
          dbg << "Considering patch "
              << "ID=" << setw(2) << left << finePatch->_patchID << " "
              << "owner=" << setw(2) << left << finePatch->_procID << " "
              << finePatch->_box << " ..." << "\n";
          // Intersection of fine and coarse patches in fine-level subscripts
          Box fineIntersect = coarseRefined.intersect(finePatch->_box);
          // Intersection of fine and coarse patches in coarse-level subscripts
          Box coarseUnderFine(fineIntersect.get(Left) / refRat, 
                              fineIntersect.get(Right) / refRat);
          dbg.setLevel(2);
          dbg << "fineIntersect   = " << fineIntersect   << "\n";
          dbg << "coarseUnderFine = " << coarseUnderFine << "\n";
          //===================================================================
          // Delete the underlying coarse cell equations (those under the
          // fine patch). Replace them by the identity operator.
          //===================================================================
          // No need to change the graph; only the matrix changes.
          //makeUnderlyingIdentity(level,stencil,coarseUnderFine);
          
          //===================================================================
          // Loop over coarse-to-fine internal boundaries of the fine patch;
          // add C-to-F connections; delete the old C-to-coarseUnderFine
          // connections.
          //===================================================================
          for (Counter d = 0; d < numDims; d++) {
            for (Side s = Left; s <= Right; ++s) {
              if (finePatch->getBoundaryType(d,s) == Patch::CoarseFine) {
               dbg << "fine boundary is " << finePatch->getBoundaryType(d,s) << "\n";
               // Coarse cell nbhring the C/F boundary on the other side
                // of the fine patch
                Box faceCoarseBox =
                  fineIntersect.coarseNbhrExtents(refRat,d,s);
                if (patch->_box.intersect(faceCoarseBox).degenerate()) {
                  // The coarse nbhrs of this fine patch C/F boundary are
                  // outside this coarse patch, ignore this face.
                  continue;
                }
                // Set the coarse-to-fine connections
                makeConnections(Graph,hier,stencil,level,finePatch,d,s,CoarseToFine);

              } // end if CoarseFine boundary
            } // end for s
          } // end for d
        } // end for all fine patches that cover this patch
      } // end if (level < numLevels-1)
    } // end for i (patches)
    dbg.unindent();
  } // end for level
  serializeProcsEnd();
  
  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  dbg << "Assembled graph, nUVentries = "
      << hypre_SStructGraphNUVEntries(_graph) << "\n";
  funcPrint("Solver::makeGraph()",FEnd);
} // end makeGraph()

void 
Solver::makeUnderlyingIdentity(const Counter level,
                               const HYPRE_SStructStencil& stencil,
                               const Box& coarseUnderFine)
  // Replace the matrix equations for the underlying coarse box
  // with the identity matrix.
{
  dbg.setLevel(2);
  dbg << "Putting identity on underlying coarse data" << "\n"
      << "coarseUnderFine " << coarseUnderFine << "\n";
  Counter stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = scinew int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) {
    entries[entry] = entry;
  }
  const Counter numCoarseCells = coarseUnderFine.volume();
  double* values    =scinew double[stencilSize * numCoarseCells];
  double* rhsValues =scinew double[numCoarseCells];

  dbg.setLevel(3);
  dbg << "Looping over cells in coarse underlying box:" 
      << "\n";
  Counter cell = 0;
  for (Box::iterator coarse_iter = coarseUnderFine.begin();
       coarse_iter != coarseUnderFine.end(); ++coarse_iter, cell++) {
    dbg.setLevel(3);
    dbg << "cell = " << cell << " " << *coarse_iter << "\n";
    
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    // Initialize the stencil values of this cell's equation to 0
    // except the central coefficient that is 1
    values[offsetValues] = 1.0;
    for (Counter entry = 1; entry < stencilSize; entry++) {
      values[offsetValues + entry] = 0.0;
    }
    // Set the corresponding RHS entry to 0.0
    rhsValues[offsetRhsValues] = 0.0;
  } // end for cell
  
  printValues(coarseUnderFine.volume(),stencilSize,values,rhsValues);
  
  // Effect the identity operator change in the Hypre structure
  // for A
  Box box(coarseUnderFine);
  dbg0 << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
  HYPRE_SStructMatrixSetBoxValues(_A, level,
                                  box.get(Left).getData(),
                                  box.get(Right).getData(),
                                  0, stencilSize, entries, values);
  dbg0 << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
  HYPRE_SStructVectorSetBoxValues(_b, level, 
                                  box.get(Left).getData(),
                                  box.get(Right).getData(),
                                  0, rhsValues);
  delete[] values;
  delete[] rhsValues;
  delete[] entries;
} // end makeUnderlyingIdentity

void
Solver::makeInteriorEquations(const Counter level,
                              const Hierarchy& hier,
                              const HYPRE_SStructGrid& grid,
                              const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeInteriorEquations~
  // Initialize the linear system equations (LHS matrix A, RHS vector b)
  // at the interior of all patches at this level.
  //_____________________________________________________________________
{
  funcPrint("Solver::makeLinearSystem()",FBegin);
  linePrint("=",50);
  dbg0 << "Adding interior equations to A, level = "
       << level << "\n";
  linePrint("=",50);
  const Counter& numDims   = _param->numDims;
  Counter stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = scinew int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) {
    entries[entry] = entry;
  }
  const Level* lev = hier._levels[level];
  const Vector<double>& h = lev->_meshSize;
  //const Vector<Counter>& resolution = lev->_resolution;
  Vector<double> offset = 0.5 * h;
  double cellVolume = h.prod();
  dbg.indent();
  for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
    dbg0 << "At patch = " << i << "\n";
    // Add equations of interior cells of this patch to A
    Patch* patch = lev->_patchList[MYID][i];
    double* values    =scinew double[stencilSize * patch->_numCells];
    double* rhsValues =scinew double[patch->_numCells];
    double* solutionValues =scinew double[patch->_numCells];
    dbg.setLevel(3);
    dbg << "Adding interior equations at Patch " << i
        << ", Extents = " << patch->_box << "\n";
    dbg << "Looping over cells in this patch:" << "\n";

    Counter cell = 0;
    for(Box::iterator iter = patch->_box.begin();
        iter != patch->_box.end(); ++iter, cell++ ) {
      dbg << "  sub = " << *iter << "\n";
      int offsetValues    = stencilSize * cell;
      int offsetRhsValues = cell;
      // Initialize the stencil values of this cell's equation to 0
      for (Counter entry = 0; entry < stencilSize; entry++) {
        values[offsetValues + entry] = 0.0;
      }

      // Compute RHS integral over the cell. Using the mid-point
      // rule, and assuming that xCell is also the centroid of the
      // cell.
      Vector<double> xCell = offset + (h * (*iter));;
      rhsValues[offsetRhsValues] = cellVolume * _param->rhs(xCell);

      // Assuming a constant initial guess
      solutionValues[offsetRhsValues] = 1234.56;
        
      // Loop over directions
      Counter entry = 1;
      for (Counter d = 0; d < numDims; d++) {
        double faceArea = cellVolume / h[d];
        for (Side s = Left; s <= Right; ++s) {
          dbg << "--- d = " << d
              << " , s = " << s
              << " , entry = " << entry
              << " ---" << "\n";
          // Compute coordinates of:
          // This cell's center: xCell
          // The neighboring's cell data point: xNbhr
          // The face crossing between xCell and xNbhr: xFace
          Vector<double> xNbhr  = xCell;
            
          xNbhr[d] += s*h[d];

          dbg << "1) xCell = " << xCell
              << ", xNbhr = " << xNbhr << "\n";

          if( (patch->getBoundaryType(d,s) == Patch::Domain) && 
              ((*iter)[d] == patch->_box.get(s)[d]) ) {
            // Cell near a domain boundary
              
            if (patch->getBC(d,s) == Patch::Dirichlet) {
              dbg << "Near Dirichlet boundary, update xNbhr"
                  << "\n";
              xNbhr[d] = xCell[d] + 0.5*s*h[d];
            } else {
              // TODO: put something in this loop?

              // Neumann B.C., xNbhr is outside the domain. We
              // assume that a, du/dn can be continously extended
              // outside the domain to make the B.C. a du/dn = rhsBC
              // meaningful using a central difference and a
              // harmonic avg of a over the line of that central
              // difference. Otherwise, go back to the Dirichlet
              // code at the expense of larger truncation errors
              // near these boundaries, that should not matter, as
              // in the Dirichlet case.
            }
                            
          } // end cell near domain boundary

          Vector<double> xFace = xCell;
          xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
          dbg << "xCell = " << xCell
              << ", xFace = " << xFace
              << ", xNbhr = " << xNbhr << "\n";

          //--- Compute flux ---
          // Harmonic average of diffusion for this face 
          double a    = _param->harmonicAvg(xCell,xNbhr,xFace); 
          double diff = fabs(xNbhr[d] - xCell[d]);  // for FD approx of flux
          double flux = a * faceArea / diff;        // total flux thru face

          // Accumulate this flux's contribution to values
          // if we are not near a C/F boundary.
          // TODO: CHECK THIS!!!
          if (!((patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
                ((*iter)[d] == patch->_box.get(s)[d]))) {
            values[offsetValues        ] += flux;
            values[offsetValues + entry] -= flux;
          }

          // If we are next to a domain boundary, eliminate boundary variable
          // from the linear system.
          if( (patch->getBoundaryType(d,s) == Patch::Domain) && 
              ((*iter)[d] == patch->_box.get(s)[d]) ) {
            // Cell near a domain boundary
            // Nbhr is at the boundary, eliminate it from values

            if (patch->getBC(d,s) == Patch::Dirichlet) {
              dbg << "Near Dirichlet boundary, eliminate nbhr, "
                  << "coef = " << values[offsetValues + entry]
                  << ", rhsBC = " << _param->rhsBC(xNbhr) << "\n";
              // Pass boundary value to RHS
              rhsValues[offsetRhsValues] -= 
                values[offsetValues + entry] * _param->rhsBC(xNbhr);

              values[offsetValues + entry] = 0.0; // Eliminate connection
              // TODO:
              // Add to rhsValues if this is a non-zero Dirichlet B.C. !!
            } else { // Neumann B.C.
              dbg << "Near Neumann boundary, eliminate nbhr"
                  << "\n";
              // TODO:
              // DO NOT ADD FLUX ABOVE, and add to rhsValues appropriately,
              // if this is a non-zero Neumann B.C.
            }
          }
          entry++;
        } // end for s
      } // end for d

        //======== BEGIN GOOD DEBUGGING CHECK =========
        // This will set the diagonal entry of this cell's equation
        // to cell so that we can compare our cell numbering with
        // Hypre's cell numbering within each patch.
        // Hypre does it like we do: first loop over x, then over y,
        // then over z.
        //        values[offsetValues] = cell;
        //======== END GOOD DEBUGGING CHECK =========

    } // end for cell
      
    printValues(patch->_box.volume(),stencilSize,
                values,rhsValues,solutionValues);

    // Add this patch's interior equations to the LHS matrix A 
    dbg0 << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
    HYPRE_SStructMatrixSetBoxValues(_A, level, 
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
                                    0, stencilSize, entries, values);

    // Add this patch's interior RHS to the RHS vector b 
    dbg0 << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
    HYPRE_SStructVectorSetBoxValues(_b, level,
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
                                    0, rhsValues);

    // Add this patch's interior initial guess to the solution vector x 
    dbg0 << "Calling HYPRE_SStructVectorSetBoxValues x" << "\n";
    HYPRE_SStructVectorSetBoxValues(_x, level,
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
                                    0, solutionValues);

    delete[] values;
    delete[] rhsValues;
    delete[] solutionValues;
  } // end for patch
  dbg.unindent();
  delete entries;
  funcPrint("Solver::makeInteriorEquations()",FEnd);
} // end makeInteriorEquations()

void
Solver::makeLinearSystem(const Hierarchy& hier,
                         const HYPRE_SStructGrid& grid,
                         const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeLinearSystem~
  // Initialize the linear system: set up the values on the links of the
  // graph of the LHS matrix A and value of the RHS vector b at all
  // patches of all levels. Delete coarse data underlying fine patches.
  //_____________________________________________________________________
{
  serializeProcsBegin();
  funcPrint("Solver::makeLinearSystem()",FBegin);
  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  serializeProcsEnd();

  //======================================================================
  // Add structured equations (stencil-based) at the interior of
  // each patch at every level to the graph.
  // Eliminate boundary conditions at domain boundaries.
  //======================================================================
  linePrint("*",50);
  dbg0 << "Matrix structured (interior) equations" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    serializeProcsBegin();
    makeInteriorEquations(level,hier,grid,stencil);
    serializeProcsEnd();
  } // end for level

  //======================================================================
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  dbg0 << "Matrix unstructured (C/F) equations" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg.setLevel(1);
    const Level* lev = hier._levels[level];
    dbg.indent();
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      dbg.setLevel(2);
      linePrint("%",40);
      dbg << "Processing Patch" << "\n"
          << *patch << "\n";
      linePrint("%",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        linePrint("=",50);
        dbg0 << "Building fine-to-coarse connections" << "\n";
        linePrint("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              makeConnections(Matrix,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        linePrint("=",50);
        dbg0 << "Building coarse-to-fine connections" 
             << " Patch ID = " << patch->_patchID
             << "\n";
        linePrint("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        dbg.indent();
        // List of fine patches covering this patch
        vector<Patch*> finePatchList = hier.finePatchesOverMe(*patch);
        Box coarseRefined(patch->_box.get(Left) * refRat,
                          (patch->_box.get(Right) + 1) * refRat - 1);
        dbg << "coarseRefined " << coarseRefined << "\n";
    
        //===================================================================
        // Loop over next-finer level patches that cover this patch
        //===================================================================
        for (vector<Patch*>::iterator iter = finePatchList.begin();
             iter != finePatchList.end(); ++iter) {
          //===================================================================
          // Compute relevant boxes at coarse and fine levels
          //===================================================================
          Patch* finePatch = *iter;
          dbg.setLevel(3);
          dbg << "Considering patch "
              << "ID=" << setw(2) << left << finePatch->_patchID << " "
              << "owner=" << setw(2) << left << finePatch->_procID << " "
              << finePatch->_box << " ..." << "\n";
          // Intersection of fine and coarse patches in fine-level subscripts
          Box fineIntersect = coarseRefined.intersect(finePatch->_box);
          // Intersection of fine and coarse patches in coarse-level subscripts
          Box coarseUnderFine(fineIntersect.get(Left) / refRat, 
                              fineIntersect.get(Right) / refRat);
          
          //===================================================================
          // Delete the underlying coarse cell equations (those under the
          // fine patch). Replace them by the identity operator.
          //===================================================================
          makeUnderlyingIdentity(level,stencil,coarseUnderFine);
          
          //===================================================================
          // Loop over coarse-to-fine internal boundaries of the fine patch;
          // add C-to-F connections; delete the old C-to-coarseUnderFine
          // connections.
          //===================================================================
          for (Counter d = 0; d < numDims; d++) {
            for (Side s = Left; s <= Right; ++s) {
              if (finePatch->getBoundaryType(d,s) == Patch::CoarseFine) {
                // Coarse cell nbhring the C/F boundary on the other side
                // of the fine patch
                Box faceCoarseBox =
                  fineIntersect.coarseNbhrExtents(refRat,d,s);
                if (patch->_box.intersect(faceCoarseBox).degenerate()) {
                  // The coarse nbhrs of this fine patch C/F boundary are
                  // outside this coarse patch, ignore this face.
                  continue;
                }
                // Set the coarse-to-fine connections
                makeConnections(Matrix,hier,stencil,level,finePatch,d,s,CoarseToFine);

              } // end if CoarseFine boundary
            } // end for s
          } // end for d
        } // end for all fine patches that cover this patch
      } // end if (level < numLevels-1)
    } // end for i (patches)
    dbg.unindent();
  } // end for level
  serializeProcsEnd();

  funcPrint("Solver::makeLinearSystem()",FEnd);
} // end makeLinearSystem()

void
Solver::printMatrix(const string& fileName /* = "solver" */)
{
  dbg << "Solver::printMatrix() begin" << "\n";
  if (!_param->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
    HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
  }
  dbg << "Solver::printMatrix() end" << "\n";
}

void
Solver::printRHS(const string& fileName /* = "solver" */)
{
  if (!_param->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _b, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_parB, (fileName + ".par").c_str());
  }
}

void
Solver::printSolution(const string& fileName /* = "solver" */)
{
  if (!_param->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _x, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_parX, (fileName + ".par").c_str());
  }
}

void
Solver::printValues(const Counter numCells,
                    const Counter stencilSize,
                    const double* values /* = 0 */,
                    const double* rhsValues /* = 0 */,
                    const double* solutionValues /* = 0 */)
  // Print values, rhsValues vectors
{
  dbg.setLevel(3);
  dbg << "--- Printing values,rhsValues,solutionValues arrays ---" << "\n";
  for (Counter cell = 0; cell < numCells; cell++) {
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    dbg << "cell = " << cell << "\n";
    if (values) {
      for (Counter entry = 0; entry < stencilSize; entry++) {
        dbg << "values   ["
            << setw(5) << left << offsetValues + entry
            << "] = " << values[offsetValues + entry] << "\n";
      }
    }
    if (rhsValues) {
      dbg << "rhsValues["
          << setw(5) << left << offsetRhsValues
          << "] = " << rhsValues[offsetRhsValues] << "\n";
    }
    if (solutionValues) {
      dbg << "solutionValues["
          << setw(5) << left << offsetRhsValues
          << "] = " << solutionValues[offsetRhsValues] << "\n";
    }
    dbg << "-------------------------------" << "\n";
  } // end for cell
} // end printValues()


std::ostream&
operator << (std::ostream& os, const Solver::CoarseFineViewpoint& v)
     // Write side s to the stream os.
{
  if      (v == Solver::CoarseToFine) os << "CoarseToFine";
  else if (v == Solver::FineToCoarse) os << "FineToCoarse";
  else os << "N/A";
  return os;
}


std::ostream&
operator << (std::ostream& os, const Solver::ConstructionStatus& s)
{
  if      (s == Solver::Graph ) os << "Graph ";
  else if (s == Solver::Matrix) os << "Matrix";
  else os << "ST WRONG!!!";
  return os;
}
