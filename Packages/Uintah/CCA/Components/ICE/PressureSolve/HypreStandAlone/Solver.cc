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
  dbg << "Solver::initialize() begin" << "\n";
  _requiresPar =
    (((_solverID >= 20) && (_solverID <= 30)) ||
     ((_solverID >= 40) && (_solverID < 60)));
  dbg0 << proc() << "requiresPar = " << _requiresPar << "\n";

  makeGraph(hier, grid, stencil);
  initializeData(hier, grid);
  makeLinearSystem(hier, grid, stencil);
  assemble();
  dbg << "Solver::initialize() end" << "\n";
}

void
Solver::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid)
{
  dbg << "Solver::initializeData() begin" << "\n";

  // Create an empty matrix with the graph non-zero pattern
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, _graph, &_A);
  dbg0 << proc() << "Created empty SStructMatrix" << "\n";
  // Initialize RHS vector b and solution vector x
  dbg << proc() << "Create empty b,x" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  dbg << proc() << "Done b" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  dbg << proc() << "Done x" << "\n";

  // If using AMG, set (A,b,x)'s object type to ParCSR now
  if (_requiresPar) {
    dbg0 << proc() << "Matrix object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
    dbg << proc() << "Vector object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    dbg << proc() << "Done b" << "\n";
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
    dbg << proc() << "Done x" << "\n";
  }
  dbg << proc() << "Init A" << "\n";
  HYPRE_SStructMatrixInitialize(_A);
  dbg << proc() << "Init b,x" << "\n";
  HYPRE_SStructVectorInitialize(_b);
  dbg << proc() << "Done b" << "\n";
  HYPRE_SStructVectorInitialize(_x);
  dbg << proc() << "Done x" << "\n";

  dbg << "Solver::initializeData() end" << "\n";
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
  dbg0 << proc() << "----------------------------------------------------" << "\n";
  dbg0 << proc() << "Solver setup phase" << "\n";
  dbg0 << proc() << "----------------------------------------------------" << "\n";
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
  dbg0 << proc() << "----------------------------------------------------" << "\n";
  dbg0 << proc() << "Solver solve phase" << "\n";
  dbg0 << proc() << "----------------------------------------------------" << "\n";

  this->solve(); // which setup will this be? The derived class's?

  //-----------------------------------------------------------
  // Gather the solution vector
  //-----------------------------------------------------------
  // TODO: SolverSStruct is derived from Solver; implement the following
  // in SolverSStruct. For SolverStruct (PFMG), another gather vector required.
  dbg0 << proc() << "----------------------------------------------------" << "\n";
  dbg0 << proc() << "Gather the solution vector" << "\n";
  dbg0 << proc() << "----------------------------------------------------" << "\n";

  HYPRE_SStructVectorGather(_x);
} //end solve()

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
  dbg << proc() << "Solver::makeGraph() begin" << "\n";
  //-----------------------------------------------------------
  // Set up the graph
  //-----------------------------------------------------------
  dbg0 << proc()
       << "----------------------------------------------------" << "\n";
  dbg0 << proc()
       << "Set up the graph" << "\n";
  dbg0 << proc()
       << "----------------------------------------------------" << "\n";
  serializeProcsEnd();

  // Create an empty graph
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &_graph);
  // If using AMG, set graph's object type to ParCSR now
  if (_requiresPar) {
    dbg0 << proc() << "graph object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();

  serializeProcsBegin();

  //======================================================================
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //======================================================================

  for (Counter level = 0; level < numLevels; level++) {
    dbg << proc() << "  Initializing graph stencil at level " << level
        << " of " << numLevels << "\n";
    HYPRE_SStructGraphSetStencil(_graph, level, 0, stencil);
  }
  
  // Add the unstructured part of the stencil connecting the coarse
  // and fine level at every C/F boundary.

  for (Counter level = 1; level < numLevels; level++) {
    dbg << proc() << "  Updating coarse-fine boundaries at level "
        << level << "\n";
    const Level* lev = hier._levels[level];
    //    const Level* coarseLev = hier._levels[level-1];
    const Vector<Counter>& refRat = lev->_refRat;

    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      dbg << proc() << "Processing patch " << patch->_box << "\n";

      // Loop over fine-to-coarse boundaries of this patch
      for (Counter d = 0; d < numDims; d++) {
        for (Side s = Left; s <= Right; ++s) {
          if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
            dbg << proc()
                << "\t--- Processing Fine-to-Coarse face d = " << d
                << " , s = " << s << " ---" << "\n";
            // Fine cells of this face
            Box faceFineBox = patch->_box.faceExtents(d,s);
            dbg << proc()
                << "\tFace(d = " << char(d+'x')
                << ", s = " << s << ") "
                << faceFineBox << "\n";

            Box faceCoarseBox = faceFineBox.coarseNbhrExtents( refRat, d, s );

            // Loop over the coarse cells that border the fine face.
            for( Box::iterator coarse_iter = faceCoarseBox.begin();
                 coarse_iter != faceCoarseBox.end(); ++coarse_iter ) {

              // Fine cell 
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
              
              Box fineCellFace( cellFaceLower, cellFaceUpper );

              // Loop over the fine cells that neighbor the coarse cell.
              for( Box::iterator fine_iter = fineCellFace.begin();
                   fine_iter != fineCellFace.end(); ++fine_iter ) {
                dbg << "Coarse cell: " << *coarse_iter << "\n";

                // Add the connections between the fine and coarse nodes
                // to the graph
                dbg << proc() << "  Adding fine-to-coarse connection to graph" << "\n";
                HYPRE_SStructGraphAddEntries(_graph,
                                             level,(*fine_iter).getData(),
                                             0,
                                             level-1,(*coarse_iter).getData(),
                                             0);

                // TODO: this call does not work when different procs
                // own the fine, coarse patches. Move to pseudo code below.
                 HYPRE_SStructGraphAddEntries(_graph,
                                              level-1,(*coarse_iter).getData(),
                                              0,
                                              level,(*fine_iter).getData(),
                                              0); 
              } // end for fine_iter
            } // end for coarse_iter
          } // end if boundary is CF interface
        } // end for s
      } // end for d

#if 0
      // Loop over coarse-to-fine internal boundaries of this patch
      // Pseudo-code
      for (all finePatch that patch intersects) {

        for (Counter d = 0; d < numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (finePatch->getBoundaryType(d,s) == Patch::CoarseFine) {

              if (coarseNbhrExtents is outside patch extent) {
                continue;
              }
              
              // Find all coarse cells that nbhr this finePatch boundary
              
              // Loop over the coarse nbhrs
              for (Counter cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,
                                   active,subCoarse,eoc)) {

                for (Counter child = 0; !eocChild;
                     child++,
                       IndexPlusPlus(zero,ref1,activeChild,subChild,eocChild)) {
                  
                  dbg << proc()
                      << "  Adding coarse-to-fine connection to graph" << "\n";
                  HYPRE_SStructGraphAddEntries(_graph,
                                               level-1, hypreSubCoarse, 0,
                                               level,   hypreSubFine,   0);
                } // end for fine cell children of the nbhr of the coarse cell
              } // end for coarse cells
            
            }
          }
        }
      }
#endif 
    } // end for i (patches)
  } // end for level
  serializeProcsEnd();

  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  dbg << proc() << "Assembled graph, nUVentries = "
      << hypre_SStructGraphNUVEntries(_graph) << "\n";
  dbg << proc() << "Solver::makeGraph() end" << "\n";
} // end makeGraph()

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
  dbg0 << proc() << "Solver::makeLinearSystem() begin" << "\n";
  serializeProcsBegin();
  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  //
  // Add equations at all interior cells of every patch owned by this proc
  // to A. Eliminate boundary conditions at domain boundaries.

  dbg0 << proc() << "Adding interior equations to A" << "\n";

  int stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];

  for (Counter entry = 0; entry < stencilSize; entry++) entries[entry] = entry;

  for (Counter level = 0; level < numLevels; level++) {
    dbg0 << proc() << "At level = level" << "\n";
    const Level* lev = hier._levels[level];
    const Vector<double>& h = lev->_meshSize;
    //const Vector<Counter>& resolution = lev->_resolution;
    Vector<double> offset = 0.5 * h;
    double cellVolume = h.prod();
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      dbg0 << proc() << "At patch = " << i << "\n";
      // Add equations of interior cells of this patch to A
      Patch* patch = lev->_patchList[MYID][i];
      double* values    = new double[stencilSize * patch->_numCells];
      double* rhsValues = new double[patch->_numCells];
      double* solutionValues = new double[patch->_numCells];

      dbg << proc() << "  Adding interior equations at Patch " << i
          << ", Extents = ";
      dbg << patch->_box << "\n";
      dbg << proc() << "Looping over cells in this patch:" << "\n";

      Counter cell = 0;
      for( Box::iterator iter = patch->_box.begin();
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
        int entry = 1;
        for (Counter d = 0; d < numDims; d++) {
          double faceArea = cellVolume / h[d];
          for (Side s = Left; s <= Right; ++s) {
            dbg << proc() << "--- d = " << d
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
                dbg << proc() << "Near Dirichlet boundary, update xNbhr"
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
                dbg << proc() << "Near Dirichlet boundary, eliminate nbhr, "
                    << "coef = " << values[offsetValues + entry]
                    << ", rhsBC = " << _param->rhsBC(xNbhr) << "\n";
                // Pass boundary value to RHS
                rhsValues[offsetRhsValues] -= 
                  values[offsetValues + entry] * _param->rhsBC(xNbhr);

                values[offsetValues + entry] = 0.0; // Eliminate connection
                // TODO:
                // Add to rhsValues if this is a non-zero Dirichlet B.C. !!
              } else { // Neumann B.C.
                dbg << proc() << "Near Neumann boundary, eliminate nbhr"
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
      
      dbg0 << proc() << "Calling HYPRE SetBoxValues()" << "\n";
      printValues(patch,stencilSize,values,rhsValues,solutionValues);

      // Add this patch's interior equations to the LHS matrix A 
      dbg0 << proc() << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
      HYPRE_SStructMatrixSetBoxValues(_A, level, 
                                      patch->_box.get(Left).getData(),
                                      patch->_box.get(Right).getData(),
                                      0, stencilSize, entries, values);

      // Add this patch's interior RHS to the RHS vector b 
      dbg0 << proc() << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
      HYPRE_SStructVectorSetBoxValues(_b, level,
                                      patch->_box.get(Left).getData(),
                                      patch->_box.get(Right).getData(),
                                      0, rhsValues);

      // Add this patch's interior initial guess to the solution vector x 
      dbg0 << proc() << "Calling HYPRE_SStructVectorSetBoxValues x" << "\n";
      HYPRE_SStructVectorSetBoxValues(_x, level,
                                      patch->_box.get(Left).getData(),
                                      patch->_box.get(Right).getData(),
                                      0, solutionValues);

      delete[] values;
      delete[] rhsValues;
      delete[] solutionValues;
    } // end for patch
  } // end for level

  delete entries;
  dbg0 << proc() << "Done adding interior equations" << "\n";

  dbg0 << proc() << "Begin C/F interface equation construction" << "\n";
  // 
  // Set the values on the graph links of the unstructured part
  // connecting the coarse and fine level at every C/F boundary.
  //

  for (Counter level = 1; level < numLevels; level++) {
    dbg0 << proc() << "  Updating coarse-fine boundaries at level "
         << level << "\n";
    const Level* lev = hier._levels[level];
    const Vector<Counter>& refRat = lev->_refRat;
    const Vector<double>& h = lev->_meshSize;
    Vector<double> fineOffset = 0.5 * h;
    double cellVolume = h.prod();

    const Level* coarseLev = hier._levels[level-1];
    const Vector<double>& coarseH = coarseLev->_meshSize;
    Vector<double> coarseOffset = 0.5 * coarseH;
    double coarseCellVolume = coarseH.prod();

    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
    
      dbg << proc() << "Patch i = " << setw(2) << left << i
          << "extends from " << patch->_box << "\n";
      
      // Compute the extents of the box [coarseilower,coarseiupper] at
      // the coarse patch that underlies fine patch i
      Vector<int> coarseilower(0,numDims);
      Vector<int> coarseiupper(0,numDims);

      Box coarseUnderFine( patch->_box.get(Left) / refRat, 
                           patch->_box.get(Right) / refRat );

      dbg << proc() << "Underlying coarse data: extends from "
          << coarseUnderFine << "\n";
      dbg << proc() << "\n";

      // Replace the matrix equations for the underlying coarse box
      // with the identity matrix.
      int stencilSize = hypre_SStructStencilSize(stencil);
      {
        int* entries = new int[stencilSize];
        for (Counter entry = 0; entry < stencilSize; entry++) {
          entries[entry] = entry;
        }

        Counter numCoarseCells = coarseUnderFine.volume();
        dbg << proc() << "# coarse cells in box = " << numCoarseCells << "\n";

        double* values    = new double[stencilSize * numCoarseCells];
        double* rhsValues = new double[numCoarseCells];
        
        dbg << proc() << "Looping over cells in coarse underlying box:" 
            << "\n";
        Counter cell = 0;
        for( Box::iterator coarse_iter = coarseUnderFine.begin();
             coarse_iter != coarseUnderFine.end(); ++coarse_iter, cell++ ) {

          dbg << proc() << "cell = " << cell
              << *coarse_iter << "\n";
          
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
        
        printValues(patch,stencilSize,
                    values,rhsValues);

        // Effect the identity operator change in the Hypre structure
        // for A
        HYPRE_SStructMatrixSetBoxValues(_A, level-1, 
                                        coarseUnderFine.get(Left).getData(),
                                        coarseUnderFine.get(Right).getData(),
                                        0, stencilSize, entries, values);
        HYPRE_SStructVectorSetBoxValues(_b, level-1, 
                                        coarseUnderFine.get(Left).getData(),
                                        coarseUnderFine.get(Right).getData(),
                                        0, rhsValues);
        delete[] values;
        delete[] rhsValues;
        delete[] entries;
      } // end identity matrix setting
      dbg << proc() << "Finished deleting underlying coarse data" << "\n";

      dbg << proc() << "Looping over C/F boundaries of Patch " << i
          << ", Level " << patch->_levelID << "\n";
      // Loop over C/F boundaries of this patch
      for (Counter d = 0; d < numDims; d++) {
        double faceArea = cellVolume / h[d];
        double coarseFaceArea = coarseCellVolume / coarseH[d];
        for (Side s = Left; s <= Right; ++s) {
          dbg << proc() << "  boundary( d = " << d
              << " , s = " << s 
              << " ) = " << Patch::boundaryTypeString
            [patch->getBoundaryType(d,s)].c_str()
              << "\n";
          if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
            dbg << proc() << "--- Processing Fine-to-Coarse face d = " << d
                << " , s = " << s << " ---" << "\n";
            // Fine cells of this face
            Box faceFineBox = patch->_box.faceExtents(d,s);

            // Coarse cell nbhring the C/F boundary on the other side of the
            // fine patch
            Box faceCoarseBox = faceFineBox.coarseNbhrExtents( refRat, d, s );

            // Loop over the coarse cells that border the fine face.
            for( Box::iterator coarse_iter = faceCoarseBox.begin();
                 coarse_iter != faceCoarseBox.end(); ++coarse_iter ) {

              // Fine cell 
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
              
              Box fineCellFace( cellFaceLower, cellFaceUpper );

              // Loop over the fine cells that neighbor the coarse cell.
              Counter child = 0;
              for( Box::iterator fine_iter = fineCellFace.begin();
                   fine_iter != fineCellFace.end(); ++fine_iter, ++child) {
                dbg << "Coarse cell: " << *coarse_iter << "\n";

                // Set the weights on the connections between the fine
                // and coarse nodes in A
                dbg << proc() << "    Adding C/F flux to matrix" << "\n";

                //====================================================
                // Add the flux from the fine child cell to the
                // coarse nbhr cell.
                //====================================================
                // Compute coordinates of:
                // This cell's center: xCell
                // The neighboring's cell data point: xNbhr
                // The face crossing between xCell and xNbhr: xFace

                Vector<double> xCell = fineOffset + (h * (*fine_iter));
                Vector<double> xNbhr = coarseOffset + (coarseH * (*coarse_iter));

                // xFace is a convex combination of xCell, xNbhr with
                // weights depending on the distances of the coarse
                // and fine meshsizes, i.e., their distances in the d
                // dimension from the C/F boundary.
                double alpha = coarseH[d] / (coarseH[d] + h[d]);
                dbg << proc() << "      alpha = " << alpha << "\n";

                Vector<double> xFace(0,numDims);

                for (Counter dd = 0; dd < numDims; dd++) {
                  xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
                }
                dbg << proc() << "      xCell = " << xCell
                    << ", xFace = " << xFace
                    << ", xNbhr = " << xNbhr << "\n";
                
                // Compute the harmonic average of the diffusion
                // coefficient
                double a    = 1.0; // Assumed constant a for now
                double diff = 0;
                for (Counter dd = 0; dd < numDims; dd++) {
                  diff += pow(xNbhr[dd] - xCell[dd],2);
                }
                diff = sqrt(diff);
                double flux = a * faceArea / diff;
                dbg << proc() << "      C/F flux = " << flux
                    << "   a = %f" << a
                    << " face = " << faceArea
                    << " diff = " << diff << "\n";
                
                // Add the C-F flux to the fine cell equation -
                // stencil part
                const int numStencilEntries = 1;
                int stencilEntries[numStencilEntries] = {0};
                double stencilValues[numStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level, (*fine_iter).getData(),
                                               0,
                                               numStencilEntries,
                                               stencilEntries,
                                               stencilValues);
                dbg << proc() << "      Done HYPRE_SStructMatrixAddToValues 1" << "\n";

                // Add the C-F flux to the fine cell equation - graph
                // part
                const int numGraphEntries = 1;
                int entry = stencilSize;
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
                dbg << proc() << "      fine equation, entry of coarse cell = "
                    << "entry " << entry << "\n";
                int graphEntries[numGraphEntries] = {entry};
                double graphValues[numGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level, (*fine_iter).getData(),     0,
                                               numGraphEntries,
                                               graphEntries,
                                               graphValues);
                dbg << proc() << "      Done HYPRE_SStructMatrixAddToValues 2"
<< "\n";

                // Subtract the C-F flux from the coarse cell equation
                // - stencil part
                const int numCoarseStencilEntries = 1;
                int coarseStencilEntries[numCoarseStencilEntries] = {0};
                double coarseStencilValues[numCoarseStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level-1, (*coarse_iter).getData(),  0,
                                               numCoarseStencilEntries,
                                               coarseStencilEntries,
                                               coarseStencilValues);
                dbg << proc() << "      Done HYPRE_SStructMatrixAddToValues 3" << "\n";

                // Subtract the C-F flux from the coarse cell equation
                // - graph part
                const int numCoarseGraphEntries = 1;
                int coarseGraphEntries[numCoarseGraphEntries] =
                  {stencilSize+child};
                dbg << proc() << "      coarse equation, entry of fine cell = "
                    << stencilSize+child << "\n";
                double coarseGraphValues[numCoarseGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level-1, (*coarse_iter).getData(), 0,
                                               numCoarseGraphEntries,
                                               coarseGraphEntries,
                                               coarseGraphValues);
                dbg << proc() << "      Done HYPRE_SStructMatrixAddToValues 4"
                    << "\n";

                if (child == 0) { // Potential source for bugs: remove
                                  // coarse-coarse flux only ONCE in the
                                  // child loop.
                  dbg << proc()
                      << "    Removing C/C flux connections from matrix"
                      << "\n";
                  //====================================================
                  // Remove the flux from the coarse nbhr to its coarse
                  // stencil nbhr that underlies the fine patch.
                  //====================================================
                  // Compute coordinates of: This cell's center: xCell
                  // The neighboring's cell data point: xNbhr The face
                  // crossing between xCell and xNbhr: xFace

                  Side s2 = Side(-s);
                  dbg << proc() << "      s = " << s
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
                  dbg << proc() << "      C/F flux = " << flux
                      << "   a = %f" << a
                      << " face = " << faceArea
                      << " diff = " << diff << "\n";
                  const int coarseNumStencilEntries = 2;
                  int coarseStencilEntries[coarseNumStencilEntries] =
                    {0, 2*d + ((-s+1)/2) + 1};
                  double coarseStencilValues[coarseNumStencilEntries] =
                    {-flux, flux};
                  HYPRE_SStructMatrixAddToValues(_A,
                                                 level-1,
                                                 (*coarse_iter).getData(), 0,
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

  dbg << proc() << "End C/F interface equation construction" << "\n";
  serializeProcsEnd();
  dbg0 << proc() << "Solver::makeLinearSystem() end" << "\n";
} // end makeLinearSystem()

void
Solver::printMatrix(const string& fileName /* = "solver" */)
{
  dbg << proc() << "Solver::printMatrix() begin" << "\n";
  if (!_param->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
    HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
  }
  dbg << proc() << "Solver::printMatrix() end" << "\n";
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
Solver::printValues(const Patch* patch,
                    const int stencilSize,
                    const double* values /* = 0 */,
                    const double* rhsValues /* = 0 */,
                    const double* solutionValues /* = 0 */)
  // Print values, rhsValues vectors
{
  const Counter numCells = patch->_box.volume();

  dbg << proc()
      << "--- Printing values,rhsValues,solutionValues arrays ---" << "\n";
  for (Counter cell = 0; cell < numCells; cell++) {
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    dbg << proc() << "cell = " << cell << "\n";
    if (values) {
      for (Counter entry = 0; entry < stencilSize; entry++) {
        dbg << proc() << "values   ["
            << setw(5) << left << offsetValues + entry
            << " = " << values[offsetValues + entry] << "\n";
      }
    }
    if (rhsValues) {
      dbg << proc() << "rhsValues["
          << setw(5) << left << offsetRhsValues
          << " = " << rhsValues[offsetRhsValues] << "\n";
    }
    if (solutionValues) {
      dbg << proc() << "solutionValues["
          << setw(5) << left << offsetRhsValues
          << " = " << solutionValues[offsetRhsValues] << "\n";
    }
    dbg << proc() << "-------------------------------" << "\n";
  } // end for cell
} // end printValues()
