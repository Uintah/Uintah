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
  Print("Solver::initialize() begin\n");
  _requiresPar =
    (((_solverID >= 20) && (_solverID <= 30)) ||
     ((_solverID >= 40) && (_solverID < 60)));
  Proc0Print("requiresPar = %d\n",_requiresPar);

  makeGraph(hier, grid, stencil);
  initializeData(hier, grid);
  makeLinearSystem(hier, grid, stencil);
  assemble();
  Print("Solver::initialize() end\n");
}

void
Solver::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid)
{
  Print("Solver::initializeData() begin\n");

  /* Create an empty matrix with the graph non-zero pattern */
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, _graph, &_A);
  Proc0Print("Created empty SStructMatrix\n");
  /* Initialize RHS vector b and solution vector x */
  Print("Create empty b,x\n");
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  Print("Done b\n");
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  Print("Done x\n");

  /* If using AMG, set (A,b,x)'s object type to ParCSR now */
  if (_requiresPar) {
    Proc0Print("Matrix object type set to HYPRE_PARCSR\n");
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
    Print("Vector object type set to HYPRE_PARCSR\n");
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    Print("Done b\n");
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
    Print("Done x\n");
  }
  Print("Init A\n");
  HYPRE_SStructMatrixInitialize(_A);
  Print("Init b,x\n");
  HYPRE_SStructVectorInitialize(_b);
  Print("Done b\n");
  HYPRE_SStructVectorInitialize(_x);
  Print("Done x\n");

  Print("Solver::initializeData() end\n");
}

void
Solver::assemble(void)
{
  Print("Solver::assemble() begin\n");
  /* Assemble the matrix - a collective call */
  HYPRE_SStructMatrixAssemble(_A); 
  HYPRE_SStructVectorAssemble(_b);
  HYPRE_SStructVectorAssemble(_x);

  /* For BoomerAMG solver: set up the linear system in ParCSR format */
  if (_requiresPar) {
    HYPRE_SStructMatrixGetObject(_A, (void **) &_parA);
    HYPRE_SStructVectorGetObject(_b, (void **) &_parB);
    HYPRE_SStructVectorGetObject(_x, (void **) &_parX);
  }
  Print("Solver::assemble() end\n");
}

void
Solver::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Solver setup phase\n");
  Proc0Print("----------------------------------------------------\n");
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
  /*-----------------------------------------------------------
   * Solver solve phase
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Solver solve phase\n");
  Proc0Print("----------------------------------------------------\n");

  this->solve(); // which setup will this be? The derived class's?

  /*-----------------------------------------------------------
   * Gather the solution vector
   *-----------------------------------------------------------*/
  // TODO: SolverSStruct is derived from Solver; implement the following
  // in SolverSStruct. For SolverStruct (PFMG), another gather vector required.
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Gather the solution vector\n");
  Proc0Print("----------------------------------------------------\n");

  HYPRE_SStructVectorGather(_x);
} //end solve()

void
Solver::makeGraph(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructStencil& stencil)
  /*_____________________________________________________________________
    Function makeGraph:
    Initialize the graph from stencils (interior equations) and C/F
    interface connections. Create Hypre graph object "_graph" on output.
    _____________________________________________________________________*/
{
  Print("Solver::makeGraph() begin\n");
  /*-----------------------------------------------------------
   * Set up the graph
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Set up the graph\n");
  Proc0Print("----------------------------------------------------\n");
  /* Create an empty graph */
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &_graph);
  /* If using AMG, set graph's object type to ParCSR now */
  if (_requiresPar) {
    Proc0Print("graph object type set to HYPRE_PARCSR\n");
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  serializeProcsBegin();
  /*
    Add structured equations (stencil-based) at the interior of
    each patch at every level to the graph.
  */
  for (Counter level = 0; level < numLevels; level++) {
    Print("  Initializing graph stencil at level %d of %d\n",level,numLevels);
    HYPRE_SStructGraphSetStencil(_graph, level, 0, stencil);
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
        for (Side s = Left; s <= Right; ++s) {
          if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {

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
              PrintNP("  subChild = ");
              printIndex(subChild);
              PrintNP("\n");
              vector<bool> active(numDims,true);
              bool eoc = false;
              vector<int> subCoarse = coarseNbhrLower;
              for (Counter cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,
                                   active,subCoarse,eoc)) {
                Print("  cell = %4d",cell);
                PrintNP("  subCoarse = ");
                printIndex(subCoarse);
                vector<int> subFine(numDims);
                /* Compute fine cell inside the fine patch from coarse
                   cell outside the fine patch */
                subFine = subCoarse;
                if (s == Left) {
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
                PrintNP("  subFine = ");
                printIndex(subFine);
                PrintNP("\n");
                Index hypreSubFine, hypreSubCoarse;
                ToIndex(subFine,&hypreSubFine,numDims);
                ToIndex(subCoarse,&hypreSubCoarse,numDims);
              
                /* Add the connections between the fine and coarse nodes
                   to the graph */
                Print("  Adding connection to graph\n");
                HYPRE_SStructGraphAddEntries(_graph,
                                             level,   hypreSubFine,   0,
                                             level-1, hypreSubCoarse, 0); 
                HYPRE_SStructGraphAddEntries(_graph,
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
  HYPRE_SStructGraphAssemble(_graph);
  Print("Assembled graph, nUVentries = %d\n",
        hypre_SStructGraphNUVEntries(_graph));
  Print("Solver::makeGraph() end\n");
} // end makeGraph()

void
Solver::makeLinearSystem(const Hierarchy& hier,
                         const HYPRE_SStructGrid& grid,
                         const HYPRE_SStructStencil& stencil)
  /*_____________________________________________________________________
    Function makeLinearSystem:
    Initialize the linear system: set up the values on the links of the
    graph of the LHS matrix A and value of the RHS vector b at all
    patches of all levels. Delete coarse data underlying fine patches.
    _____________________________________________________________________*/
{
  Proc0Print("Solver::makeLinearSystem() begin\n");
  serializeProcsBegin();
  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  /*
    Add equations at all interior cells of every patch owned by this proc
    to A. Eliminate boundary conditions at domain boundaries.
  */
  Proc0Print("Adding interior equations to A\n");
  vector<double> xCell(numDims);
  vector<double> xNbhr(numDims);
  vector<double> xFace(numDims);
  vector<double> offset(numDims);
  vector<double> coarseOffset(numDims);
  vector<int> sub;
  vector<bool> active(numDims);
  int stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) entries[entry] = entry;
  for (Counter level = 0; level < numLevels; level++) {
    Proc0Print("At level = %d\n",level);
    const Level* lev = hier._levels[level];
    const vector<double>& h = lev->_meshSize;
    //const vector<Counter>& resolution = lev->_resolution;
    scalarMult(h,0.5,offset);
    double cellVolume = prod(h);
    for (Counter i = 0; i < lev->_patchList.size(); i++) {
      Proc0Print("At patch = %d\n",i);
      /* Add equations of interior cells of this patch to A */
      Patch* patch = lev->_patchList[i];
      double* values    = new double[stencilSize * patch->_numCells];
      double* rhsValues = new double[patch->_numCells];
      double* solutionValues = new double[patch->_numCells];
      Index hypreilower, hypreiupper;
      ToIndex(patch->_ilower,&hypreilower,numDims);
      ToIndex(patch->_iupper,&hypreiupper,numDims);
      Print("  Adding interior equations at Patch %d, Extents = ",i);
      printIndex(patch->_ilower);
      PrintNP(" to ");
      printIndex(patch->_iupper);
      PrintNP("\n");
      Print("Looping over cells in this patch:\n");
      sub = patch->_ilower;        
      for (Counter d = 0; d < numDims; d++) active[d] = true;
      bool eoc = false;
      for (Counter cell = 0; !eoc;
           cell++,
             IndexPlusPlus(patch->_ilower,patch->_iupper,active,sub,eoc)) {
        Print("cell = %4d",cell);
        PrintNP("  sub = ");
        printIndex(sub);
        PrintNP("\n");
        int offsetValues    = stencilSize * cell;
        int offsetRhsValues = cell;
        /* Initialize the stencil values of this cell's equation to 0 */
        for (Counter entry = 0; entry < stencilSize; entry++) {
          values[offsetValues + entry] = 0.0;
        }

        /* Compute RHS integral over the cell. Using the mid-point
           rule, and assuming that xCell is also the centroid of the
           cell.
        */
        rhsValues[offsetRhsValues] = cellVolume * _param->rhs(xCell);

        /* Assuming a constant initial guess */
        solutionValues[offsetRhsValues] = 1234.56;
        
        /* Loop over directions */
        int entry = 1;
        for (Counter d = 0; d < numDims; d++) {
          double faceArea = cellVolume / h[d];
          for (Side s = Left; s <= Right; ++s) {
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

            if ((patch->getBoundaryType(d,s) == Patch::Domain) && 
                ((s == Left ) && (sub[d] == patch->_ilower[d]) ||
                ((s == Right) && (sub[d] == patch->_iupper[d])))) {
              /* Cell near a domain boundary */
              
              if (patch->getBC(d,s) == Patch::Dirichlet) {
                Print("Near Dirichlet boundary, update xNbhr\n");
                xNbhr[d] = xCell[d] + 0.5*s*h[d];
              } else {
                /* Neumann B.C., xNbhr is outside the domain. We
                assume that a, du/dn can be continously extended
                outside the domain to make the B.C. a du/dn = rhsBC
                meaningful using a central difference and a harmonic
                avg of a over the line of that central
                difference. Otherwise, go back to the Dirichlet code
                at the expense of larger truncation errors near these
                boundaries, that should not matter, as in the
                Dirichlet case.
                */
              }
                            
            } // end cell near domain boundary

            xFace    = xCell;
            xFace[d] = 0.5*(xCell[d] + xNbhr[d]);

            Print("xCell = ");
            printIndex(xCell);
            PrintNP(" xNbhr = ");
            printIndex(xNbhr);
            PrintNP(" xFace = ");
            printIndex(xFace);
            PrintNP("\n");

            /*--- Compute flux ---*/
            /* Harmonic average of diffusion for this face */
            double a    = _param->harmonicAvg(xCell,xNbhr,xFace); 
            double diff = fabs(xNbhr[d] - xCell[d]);  // for FD approx of flux
            double flux = a * faceArea / diff;        // total flux thru face

            /* Accumulate this flux's contribution to values
               if we are not near a C/F boundary. */
            if (!(((patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
                   (((s == Left ) && (sub[d] == patch->_ilower[d])) ||
                    ((s == Right) && (sub[d] == patch->_iupper[d])))))) {
              values[offsetValues        ] += flux;
              values[offsetValues + entry] -= flux;
            }

            /* If we are next to a domain boundary, eliminate boundary variable
               from the linear system. */
            if ((patch->getBoundaryType(d,s) == Patch::Domain) && 
                ((s == Left ) && (sub[d] == patch->_ilower[d]) ||
                ((s == Right) && (sub[d] == patch->_iupper[d])))) {
              /* Cell near a domain boundary */
              /* Nbhr is at the boundary, eliminate it from values */

              if (patch->getBC(d,s) == Patch::Dirichlet) {
                Print("Near Dirichlet boundary, eliminate nbhr, coef = %f, rhsBC = %f\n",
                      values[offsetValues + entry],_param->rhsBC(xNbhr));
                /* Pass boundary value to RHS */
                rhsValues[offsetRhsValues] -= 
                  values[offsetValues + entry] * _param->rhsBC(xNbhr);

                values[offsetValues + entry] = 0.0; // Eliminate connection
                // TODO:
                // Add to rhsValues if this is a non-zero Dirichlet B.C. !!
              } else { // Neumann B.C.
                Print("Near Neumann boundary, eliminate nbhr\n");
                // TODO:
                // DO NOT ADD FLUX ABOVE, and add to rhsValues appropriately,
                // if this is a non-zero Neumann B.C.
              }
            }
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
      
      Proc0Print("Calling HYPRE SetBoxValues()\n");
      printValues(patch,stencilSize,patch->_numCells,
                  values,rhsValues,solutionValues);

      /* Add this patch's interior equations to the LHS matrix A */
      Proc0Print("Calling HYPRE_SStructMatrixSetBoxValues A\n");
      HYPRE_SStructMatrixSetBoxValues(_A, level, 
                                      hypreilower, hypreiupper, 0,
                                      stencilSize, entries, values);

      /* Add this patch's interior RHS to the RHS vector b */
      Proc0Print("Calling HYPRE_SStructVectorSetBoxValues b\n");
      HYPRE_SStructVectorSetBoxValues(_b, level,
                                      hypreilower, hypreiupper, 0, 
                                      rhsValues);

      /* Add this patch's interior initial guess to the solution vector x */
      Proc0Print("Calling HYPRE_SStructVectorSetBoxValues x\n");
      HYPRE_SStructVectorSetBoxValues(_x, level,
                                      hypreilower, hypreiupper, 0, 
                                      solutionValues);

      delete[] values;
      delete[] rhsValues;
      delete[] solutionValues;
    } // end for patch
  } // end for level

  delete entries;
  Proc0Print("Done adding interior equations\n");

  Proc0Print("Begin C/F interface equation construction\n");
  /* 
     Set the values on the graph links of the unstructured part
     connecting the coarse and fine level at every C/F boundary.
  */

  for (Counter level = 1; level < numLevels; level++) {
    Proc0Print("  Updating coarse-fine boundaries at level %d\n",level);
    const Level* lev = hier._levels[level];
    const vector<Counter>& refRat = lev->_refRat;
    const vector<double>& h = lev->_meshSize;
    scalarMult(h,0.5,offset);
    double cellVolume = prod(h);

    const Level* coarseLev = hier._levels[level-1];
    const vector<double>& coarseH = coarseLev->_meshSize;
    scalarMult(coarseH,0.5,coarseOffset);
    double coarseCellVolume = prod(coarseH);

    /* Loop over patches of this proc */
    for (Counter i = 0; i < lev->_patchList.size(); i++) {
      Patch* patch = lev->_patchList[i];
    
      Print("Patch i = %2d:           extends from ",i);
      printIndex(patch->_ilower);
      PrintNP(" to ");
      printIndex(patch->_iupper);
      PrintNP("\n");
      
      /* Compute the extents of the box [coarseilower,coarseiupper] at
         the coarse patch that underlies fine patch i */
      vector<int> coarseilower(numDims);
      vector<int> coarseiupper(numDims);
      pointwiseDivide(patch->_ilower,refRat,coarseilower);
      pointwiseDivide(patch->_iupper,refRat,coarseiupper);
      Print("Underlying coarse data: extends from ");
      printIndex(coarseilower);
      PrintNP(" to ");
      printIndex(coarseiupper);
      PrintNP("\n");

      /* Replace the matrix equations for the underlying coarse box
         with the identity matrix. */
      int stencilSize = hypre_SStructStencilSize(stencil);
      {
        int* entries = new int[stencilSize];
        for (Counter entry = 0; entry < stencilSize; entry++)
          entries[entry] = entry;
        Index hyprecoarseilower, hyprecoarseiupper;
        ToIndex(coarseilower,&hyprecoarseilower,numDims);
        ToIndex(coarseiupper,&hyprecoarseiupper,numDims);
        int numCoarseCells = 1;
        for (Counter dd = 0; dd < numDims; dd++) {
          numCoarseCells *= (coarseiupper[dd] - coarseilower[dd] + 1);
        }
        Print("# coarse cells in box = %d\n",numCoarseCells);
        double* values    = new double[stencilSize * numCoarseCells];
        double* rhsValues = new double[numCoarseCells];
        
        Print("Looping over cells in coarse underlying box:\n");
        vector<int> sub = coarseilower;
        vector<bool> active(numDims);
        for (Counter d = 0; d < numDims; d++) active[d] = true;
        bool eoc = false;
        for (Counter cell = 0; !eoc;
             cell++,
               IndexPlusPlus(coarseilower,coarseiupper,active,sub,eoc)) {
          Print("cell = %4d",cell);
          PrintNP("  sub = ");
          printIndex(sub);
          PrintNP("\n");
          
          int offsetValues    = stencilSize * cell;
          int offsetRhsValues = cell;
          /* Initialize the stencil values of this cell's equation to 0
             except the central coefficient that is 1 */
          values[offsetValues] = 1.0;
          for (Counter entry = 1; entry < stencilSize; entry++) {
            values[offsetValues + entry] = 0.0;
          }
          // Set the corresponding RHS entry to 0.0
          rhsValues[offsetRhsValues] = 0.0;
        } // end for cell
        
        printValues(patch,stencilSize,numCoarseCells,
                    values,rhsValues);

        /* Effect the identity operator change in the Hypre structure
           for A */
        HYPRE_SStructMatrixSetBoxValues(_A, level-1, 
                                        hyprecoarseilower, hyprecoarseiupper,
                                        0,
                                        stencilSize, entries, values);
        HYPRE_SStructVectorSetBoxValues(_b, level-1, 
                                        hyprecoarseilower, hyprecoarseiupper,
                                        0, rhsValues);
        delete[] values;
        delete[] rhsValues;
        delete[] entries;
      } // end identity matrix setting
      Print("Finished deleting underlying coarse data\n");

      Print("Looping over C/F boundaries of Patch %d, Level %d\n",
            i,patch->_levelID);
      /* Loop over C/F boundaries of this patch */
      for (Counter d = 0; d < numDims; d++) {
        double faceArea = cellVolume / h[d];
        double coarseFaceArea = coarseCellVolume / coarseH[d];
        for (Side s = Left; s <= Right; ++s) {
          Print("  boundary( d = %d , s = %+d ) = %s\n",
                d,s,
                Patch::boundaryTypeString[patch->getBoundaryType(d,s)].c_str());
          if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {

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
              PrintNP("  subChild = ");
              printIndex(subChild);
              PrintNP("\n");
              vector<bool> active(numDims,true);
              bool eoc = false;
              vector<int> subCoarse = coarseNbhrLower;
              for (Counter cell = 0; !eoc;
                   cell++,
                     IndexPlusPlus(coarseNbhrLower,coarseNbhrUpper,active,
                                   subCoarse,eoc)) {
                Print("  cell = %4d",cell);
                PrintNP("  subCoarse = ");
                printIndex(subCoarse);
                vector<int> subFine(numDims);
                /* Compute fine cell inside the fine patch from coarse
                   cell outside the fine patch */
                subFine = subCoarse;
                if (s == Left) {
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
                PrintNP("  subFine = ");
                printIndex(subFine);
                PrintNP("\n");
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
                pointwiseMult(subFine,h,xCell);
                pointwiseAdd(xCell,offset,xCell);   
                
                pointwiseMult(subCoarse,coarseH,xNbhr);
                pointwiseAdd(xNbhr,coarseOffset,xNbhr);    
                /* xFace is a convex combination of xCell, xNbhr with
                   weights depending on the distances of the coarse
                   and fine meshsizes, i.e., their distances in the d
                   dimension from the C/F boundary. */
                double alpha = coarseH[d] / (coarseH[d] + h[d]);
                Print("      alpha = %lf\n",alpha);
                for (Counter dd = 0; dd < numDims; dd++) {
                  xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
                }
                Print("      xCell = ");
                printIndex(xCell);
                PrintNP(" xNbhr = ");
                printIndex(xNbhr);
                PrintNP(" xFace = ");
                printIndex(xFace);
                PrintNP("\n");

                /* Compute the harmonic average of the diffusion
                   coefficient */
                double a    = 1.0; // Assumed constant a for now
                double diff = 0;
                for (Counter dd = 0; dd < numDims; dd++) {
                  diff += pow(xNbhr[dd] - xCell[dd],2);
                }
                diff = sqrt(diff);
                double flux = a * faceArea / diff;
                Print("      C/F flux = %f   a = %f  face = %f  diff = %f\n",
                      flux,a,faceArea,diff);
                
                /* Add the C-F flux to the fine cell equation -
                   stencil part */
                const int numStencilEntries = 1;
                int stencilEntries[numStencilEntries] = {0};
                double stencilValues[numStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level, hypreSubFine,     0,
                                               numStencilEntries,
                                               stencilEntries,
                                               stencilValues);
                Print("      Done HYPRE_SStructMatrixAddToValues 1\n");

                /* Add the C-F flux to the fine cell equation - graph
                   part */
                const int numGraphEntries = 1;
                int entry = stencilSize;
                /* Find the correct graph entry corresponding to this coarse
                   nbhr (subCoarse) of the fine cell subFine */
                for (Counter dd = 0; dd < d; dd++) {
                  /* Are we near the left boundary and is it a C/F bdry? */
                  Side ss = Left;
                  if ((patch->getBoundaryType(dd,ss) == Patch::CoarseFine) &&
                      (subFine[dd] == patch->_ilower[dd])) {
                    entry++;
                  }
                  /* Are we near the right boundary and is it a C/F bdry? */
                  ss = Right;
                  if ((patch->getBoundaryType(dd,ss) == Patch::CoarseFine) &&
                      (subFine[dd] == patch->_iupper[dd])) {
                    entry++;
                  }
                }
                if ((s == Right) &&
                    (patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
                    (subFine[d] == patch->_ilower[d])) {
                  entry++;
                }
                Print("      fine equation, entry of coarse cell = %d\n",
                      entry);
                int graphEntries[numGraphEntries] = {entry};
                double graphValues[numGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level, hypreSubFine,     0,
                                               numGraphEntries,
                                               graphEntries,
                                               graphValues);
                Print("      Done HYPRE_SStructMatrixAddToValues 2\n");

                /* Subtract the C-F flux from the coarse cell equation
                   - stencil part */
                const int numCoarseStencilEntries = 1;
                int coarseStencilEntries[numCoarseStencilEntries] = {0};
                double coarseStencilValues[numCoarseStencilEntries] = {flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level-1, hypreSubCoarse,  0,
                                               numCoarseStencilEntries,
                                               coarseStencilEntries,
                                               coarseStencilValues);
                Print("      Done HYPRE_SStructMatrixAddToValues 3\n");

                /* Subtract the C-F flux from the coarse cell equation
                   - graph part */
                const int numCoarseGraphEntries = 1;
                int coarseGraphEntries[numCoarseGraphEntries] =
                  {stencilSize+child};
                Print("      coarse equation, entry of fine cell = %d\n",
                      stencilSize+child);
                double coarseGraphValues[numCoarseGraphEntries] = {-flux};
                HYPRE_SStructMatrixAddToValues(_A,
                                               level-1, hypreSubCoarse,     0,
                                               numCoarseGraphEntries,
                                               coarseGraphEntries,
                                               coarseGraphValues);
                Print("      Done HYPRE_SStructMatrixAddToValues 4\n");

                if (child == 0) { /* Potential source for bugs: remove
                                     coarse-coarse flux only ONCE in the
                                     child loop. */
                  Print("    Removing C/C flux connections from matrix\n");
                  /*====================================================
                    Remove the flux from the coarse nbhr to its coarse
                    stencil nbhr that underlies the fine patch.
                    ====================================================*/
                  /* Compute coordinates of: This cell's center: xCell
                     The neighboring's cell data point: xNbhr The face
                     crossing between xCell and xNbhr: xFace
                  */
                  Side s2 = Side(-s);
                  Print("      s = %d , s2 = %d\n",s,s2);
                  pointwiseMult(subCoarse,coarseH,xCell);
                  pointwiseAdd(xCell,coarseOffset,xCell);               
                  xNbhr    = xCell;
                  xNbhr[d] += s2*coarseH[d];
                  xFace    = xCell;
                  xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
                  Print("      xCell = ");
                  printIndex(xCell);
                  PrintNP(" xNbhr = ");
                  printIndex(xNbhr);
                  PrintNP(" xFace = ");
                  printIndex(xFace);
                  PrintNP("\n");

                  /* Compute the harmonic average of the diffusion
                     coefficient */
                  double a    = 1.0; // Assumed constant a for now
                  double diff = fabs(xNbhr[d] - xCell[d]);
                  double flux = a * coarseFaceArea / diff;
                  Print("      C/C flux = %f   a = %f  face = %f  diff = %f\n",
                        flux,a,coarseFaceArea,diff);

                  const int coarseNumStencilEntries = 2;
                  int coarseStencilEntries[coarseNumStencilEntries] =
                    {0, 2*d + ((-s+1)/2) + 1};
                  double coarseStencilValues[coarseNumStencilEntries] =
                    {-flux, flux};
                  HYPRE_SStructMatrixAddToValues(_A,
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

  Print("End C/F interface equation construction\n");
  serializeProcsEnd();
  Proc0Print("Solver::makeLinearSystem() end\n");
} // end makeLinearSystem()

void
Solver::printMatrix(const string& fileName /* = "solver" */)
{
  Print("Solver::printMatrix() begin\n");
  if (!_param->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
    /* Print CSR matrix in IJ format, base 1 for rows and cols */
    HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
  }
  Print("Solver::printMatrix() end\n");
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
                    const int numCells,
                    const double* values /* = 0 */,
                    const double* rhsValues /* = 0 */,
                    const double* solutionValues /* = 0 */)
  /* Print values, rhsValues vectors */
{
#if DRIVER_DEBUG
  Print("--- Printing values,rhsValues,solutionValues arrays ---\n");
  for (Counter cell = 0; cell < numCells; cell++) {
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    Print("cell = %4d\n",cell);
    if (values) {
      for (Counter entry = 0; entry < stencilSize; entry++) {
        Print("values   [%5d] = %+.3f\n",
              offsetValues + entry,values[offsetValues + entry]);
      }
    }
    if (rhsValues) {
      Print("rhsValues[%5d] = %+.3f\n",
            offsetRhsValues,rhsValues[offsetRhsValues]);
    }
    if (solutionValues) {
      Print("solutionValues[%5d] = %+.3f\n",
            offsetRhsValues,solutionValues[offsetRhsValues]);
    }
    Print("-------------------------------\n");
  } // end for cell
#endif
} // end printValues()


HypreGenericSolver::SolverType
HypreGenericSolver::solverFromTitle(const string& solverTitle)
{
  /* Determine solver type from title */
  if ((solverTitle == "SMG") ||
      (solverTitle == "smg")) {
    return SMG;
  } else if ((solverTitle == "PFMG") ||
             (solverTitle == "pfmg")) {
    return PFMG;
  } else if ((solverTitle == "SparseMSG") ||
             (solverTitle == "sparsemsg")) {
    return SparseMSG;
  } else if ((solverTitle == "CG") ||
             (solverTitle == "cg") ||
             (solverTitle == "PCG") ||
             (solverTitle == "conjugategradient")) {
    return CG;
  } else if ((solverTitle == "Hybrid") ||
             (solverTitle == "hybrid")) {
    return Hybrid;
  } else if ((solverTitle == "GMRES") ||
             (solverTitle == "gmres")) {
    return GMRES;
  } else if ((solverTitle == "AMG") ||
             (solverTitle == "amg") ||
             (solverTitle == "BoomerAMG") ||
             (solverTitle == "boomeramg")) {
    return AMG;
  } else if ((solverTitle == "FAC") ||
             (solverTitle == "fac")) {
    return FAC;
  } else {
    throw InternalError("Unknown solver type: "+solverTitle,
                        __FILE__, __LINE__);
  } // end "switch" (solverTitle)
} // end solverFromTitle()

HypreGenericSolver::precondType
HypreGenericSolver::precondFromTitle(const string& precondTitle)
{
  /* Determine preconditioner type from title */
  if ((precondTitle == "SMG") ||
      (precondTitle == "smg")) {
    return PrecondSMG;
  } else if ((precondTitle == "PFMG") ||
             (precondTitle == "pfmg")) {
    return PrecondPFMG;
  } else if ((precondTitle == "SparseMSG") ||
             (precondTitle == "sparsemsg")) {
    return PrecondSparseMSG;
  } else if ((precondTitle == "Jacobi") ||
             (precondTitle == "jacobi")) {
    return PrecondJacobi;
  } else if ((precondTitle == "Diagonal") ||
             (precondTitle == "diagonal")) {
    return PrecondDiagonal;
  } else if ((precondTitle == "AMG") ||
             (precondTitle == "amg") ||
             (precondTitle == "BoomerAMG") ||
             (precondTitle == "boomeramg")) {
    return PrecondAMG;
  } else if ((precondTitle == "FAC") ||
             (precondTitle == "fac")) {
    return PrecondFAC;
  } else {
    throw InternalError("Unknown preconditionertype: "+precondTitle,
                        __FILE__, __LINE__);
  } // end "switch" (precondTitle)
} // end precondFromTitle()

HypreDriver::Interface
HypreGenericSolver::solverInterface(const SolverType& solverType)
  /* Determine the Hypre interface this solver uses */
{
  switch (solverType) {
  case SMG:
    {
      return HypreDriver::Struct;
    }
  case PFMG:
    {
      return HypreDriver::Struct;
    }
  case SparseMSG:
    {
      return HypreDriver::Struct;
    }
  case CG:
    {
      return HypreDriver::Struct;
    }
  case Hybrid: 
    {
      return HypreDriver::Struct;
    }
  case GMRES:
    {
      return HypreDriver::Struct;
    }
  case FAC:
    {
      return HypreDriver::SStruct;
    }
  case AMG:
    {
      return HypreDriver::ParCSR;
    }
  default:
    throw InternalError("Unsupported solver type: "+solverType,
                        __FILE__, __LINE__);
  } // switch (solverType)
} // end solverInterface()
